/* FacePass PAM bridge. Authentication decisions come only from the root broker.
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <security/pam_modules.h>
#include <security/pam_appl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/random.h>
#include <poll.h>
#include <pwd.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FACEPASS_SOCKET
#define FACEPASS_SOCKET "/run/facepass/auth.sock"
#endif
#ifndef FACEPASS_PEER_UID
#define FACEPASS_PEER_UID 0
#endif
#ifndef FACEPASS_AUTH_TIMEOUT
#define FACEPASS_AUTH_TIMEOUT 47
#endif

static double monotime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

static int converse(pam_handle_t *pamh, int style, const char *text, char **answer) {
    const void *item = NULL;
    if (pam_get_item(pamh, PAM_CONV, &item) != PAM_SUCCESS || !item) return PAM_CONV_ERR;
    const struct pam_conv *conv = item;
    struct pam_message message = {style, text};
    const struct pam_message *messages[] = {&message};
    struct pam_response *response = NULL;
    int status = conv->conv(1, messages, &response, conv->appdata_ptr);
    if (status != PAM_SUCCESS) return status;
    if (answer) *answer = response ? response[0].resp : NULL;
    else if (response && response[0].resp) {
        explicit_bzero(response[0].resp, strlen(response[0].resp));
        free(response[0].resp);
    }
    free(response);
    return PAM_SUCCESS;
}

static int connect_broker(void) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(FACEPASS_SOCKET) >= sizeof(address.sun_path)) { close(fd); return -1; }
    strcpy(address.sun_path, FACEPASS_SOCKET);
    int result = connect(fd, (struct sockaddr *)&address, sizeof(address));
    if (result < 0) {
        struct pollfd waitfd = {fd, POLLOUT, 0};
        int error = 0;
        socklen_t size = sizeof(error);
        if (errno != EINPROGRESS || poll(&waitfd, 1, 400) != 1 ||
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) || error) { close(fd); return -1; }
    }
    struct ucred peer;
    socklen_t size = sizeof(peer);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) || peer.uid != FACEPASS_PEER_UID) {
        close(fd); return -1;
    }
    return fd;
}

static int read_line(int fd, char *line, size_t length, double deadline) {
    size_t used = 0;
    while (monotime() < deadline && used + 1 < length) {
        struct pollfd waitfd = {fd, POLLIN, 0};
        int result = poll(&waitfd, 1, 100);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) return -1;
        if (!result) continue;
        char character;
        ssize_t count = read(fd, &character, 1);
        if (count < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (count != 1 || (unsigned char)character > 126 || (character != '\n' && character < 32)) return -1;
        if (character == '\n') { line[used] = '\0'; return 0; }
        line[used++] = character;
    }
    return -1;
}

static int request_broker(pam_handle_t *pamh, const char *operation, uid_t uid,
                          const char *service, const char *nonce) {
    int fd = connect_broker();
    if (fd < 0) return PAM_AUTHINFO_UNAVAIL;
    char request[256], line[512], expected[64];
    int length = snprintf(request, sizeof(request), "%s %u %s %s\n", operation, uid, service, nonce);
    if (length <= 0 || length >= (int)sizeof(request) || send(fd, request, (size_t)length, MSG_NOSIGNAL) != length) {
        close(fd); return PAM_AUTHINFO_UNAVAIL;
    }
    double deadline = monotime() + (!strcmp(operation, "STATUS") ? 2 : FACEPASS_AUTH_TIMEOUT);
    int status = PAM_AUTH_ERR;
    snprintf(expected, sizeof(expected), "%s %s", !strcmp(operation, "STATUS") ? "ENROLLED" : "SUCCESS", nonce);
    for (int messages = 0; messages < 64 && !read_line(fd, line, sizeof(line), deadline); messages++) {
        if (!strcmp(line, expected)) { status = PAM_SUCCESS; break; }
        if (!strncmp(line, "INFO ", 5) && strcmp(operation, "STATUS")) {
            if (converse(pamh, PAM_TEXT_INFO, line + 5, NULL) != PAM_SUCCESS) break;
        } else {
            snprintf(expected, sizeof(expected), "UNAVAILABLE %s", nonce);
            if (!strcmp(line, expected)) status = PAM_AUTHINFO_UNAVAIL;
            break;
        }
    }
    close(fd);
    return status;
}

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *username = NULL;
    const void *item = NULL;
    int face_only = argc == 1 && !strcmp(argv[0], "face_only");
    if (argc > 1 || (argc == 1 && !face_only)) return PAM_AUTH_ERR;
    if (flags & PAM_SILENT) return PAM_AUTHINFO_UNAVAIL;
    if (pam_get_user(pamh, &username, NULL) != PAM_SUCCESS || !username ||
        pam_get_item(pamh, PAM_SERVICE, &item) != PAM_SUCCESS || !item) return PAM_AUTH_ERR;
    const char *service = item;
    if (strcmp(service, "kde") && strcmp(service, "kde-face") && strcmp(service, "plasmalogin") && strcmp(service, "sddm")) return PAM_IGNORE;
    if (!face_only && pam_get_item(pamh, PAM_AUTHTOK, &item) == PAM_SUCCESS && item && ((const char *)item)[0])
        return PAM_IGNORE;
    struct passwd record, *user = NULL;
    char userbuffer[16384];
    if (getpwnam_r(username, &record, userbuffer, sizeof(userbuffer), &user) || !user || user->pw_uid < 1000)
        return face_only ? PAM_AUTHINFO_UNAVAIL : PAM_IGNORE;
    unsigned char random[16];
    size_t count = 0;
    while (count < sizeof(random)) {
        ssize_t result = getrandom(random + count, sizeof(random) - count, 0);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return PAM_AUTHINFO_UNAVAIL;
        count += (size_t)result;
    }
    char nonce[33];
    for (size_t i = 0; i < sizeof(random); i++) snprintf(nonce + i * 2, 3, "%02x", random[i]);
    if (request_broker(pamh, "STATUS", user->pw_uid, service, nonce) != PAM_SUCCESS)
        return face_only ? PAM_AUTHINFO_UNAVAIL : PAM_IGNORE;
    if (!face_only) {
        char *answer = NULL;
        if (converse(pamh, PAM_TEXT_INFO, "FacePass is ready. Use your password, or choose FacePass / submit an empty password.", NULL) != PAM_SUCCESS)
            return PAM_CONV_ERR;
        if (converse(pamh, PAM_PROMPT_ECHO_OFF, "Password (empty for FacePass): ", &answer) != PAM_SUCCESS || !answer)
            return PAM_CONV_ERR;
        if (answer[0]) {
            int status = pam_set_item(pamh, PAM_AUTHTOK, answer);
            explicit_bzero(answer, strlen(answer)); free(answer);
            return status == PAM_SUCCESS ? PAM_IGNORE : PAM_AUTH_ERR;
        }
        free(answer);
    }
    return request_broker(pamh, "AUTH", user->pw_uid, service, nonce);
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}
