"""Small, reversible KDE/PAM edits, shared by installer and offline tests."""

import re

PAM_BLOCK = """# BEGIN FACEPASS
auth [success=done ignore=ignore default=die] /usr/lib64/security/pam_facepass.so
# END FACEPASS
"""
BUTTON = """    // BEGIN FACEPASS
    property bool facePassReady: authenticator.infoMessage.startsWith("FacePass is ready.")
    property bool facePassRunning: false

    property Connections facePassConnections: Connections {
        target: authenticator
        function onInfoMessageChanged() {
            if (authenticator.infoMessage.startsWith("FacePass is ready.")) {
                sessionManager.facePassReady = true;
                sessionManager.facePassRunning = false;
            }
        }
        function onFailed(kind) {
            if (kind === 0) sessionManager.facePassRunning = false;
        }
    }

    PlasmaComponents3.Button {
        objectName: "facePassButton"
        text: sessionManager.facePassRunning ? "Checking face..." : "FacePass"
        icon.name: "camera-web"
        Layout.alignment: Qt.AlignHCenter
        visible: sessionManager.facePassReady
        enabled: !authenticator.graceLocked && !sessionManager.facePassRunning
        onClicked: {
            sessionManager.facePassRunning = true;
            passwordBox.text = "";
            sessionManager.startLogin();
        }
    }
    // END FACEPASS
"""
MESSAGE = """        // BEGIN FACEPASS
        if (msg.startsWith("FacePass")) {
            root.notification = msg;
            return;
        }
        // END FACEPASS
"""


def remove_block(text, style="pam"):
    prefix = "#" if style == "pam" else "//"
    return re.sub(r"(?m)^[ \t]*" + prefix + r" BEGIN FACEPASS\n.*?^[ \t]*" + prefix
                  + r" END FACEPASS\n", "", text, flags=re.DOTALL)


def patch_pam(text):
    if "# BEGIN FACEPASS" in text:
        if PAM_BLOCK not in text:
            raise ValueError("unknown existing FacePass PAM configuration")
        return text
    if not re.search(r"(?m)^auth\s+substack\s+password-auth\s*$", text):
        raise ValueError("unsupported PAM configuration: expected Fedora password-auth")
    # Place before every existing auth rule, keeping all account/session rules.
    line = re.search(r"(?m)^auth\s", text)
    return text[:line.start()] + PAM_BLOCK + text[line.start():]


def patch_main(text):
    if "// BEGIN FACEPASS" in text:
        return text
    anchor = "    component FailableLabel : PlasmaComponents3.Label {"
    if text.count(anchor) != 1 or "function startLogin()" not in text:
        raise ValueError("unsupported KDE lock-screen layout")
    return text.replace(anchor, BUTTON + anchor)


def patch_messages(text):
    if "// BEGIN FACEPASS" in text:
        return text
    anchor = "    function handleMessage(msg) {\n"
    if text.count(anchor) != 1:
        raise ValueError("unsupported KDE lock-screen messages")
    return text.replace(anchor, anchor + MESSAGE)
