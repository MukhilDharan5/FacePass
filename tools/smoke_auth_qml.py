"""Load the actual patched KDE component and exercise its visible FacePass button."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tools.auth_integration import patch_main, patch_messages

TEST = '''import QtQuick
import QtTest
import "." as Local

Item {
    id: root
    width: 800; height: 700
    signal clearPassword()
    signal notificationRepeated()
    property string notification: ""
    function i18ndc(domain, context, message) { return message; }
    QtObject {
        id: authenticator
        property string infoMessage: ""
        property bool graceLocked: false
        property int authenticatorTypes: 0
        signal failed(int kind)
        signal noninteractiveError(int kind, var authenticator)
    }
    Local.MainBlock {
        id: block
        anchors.fill: parent
        lockScreenUiVisible: true
        showUserList: false
    }
    SignalSpy { id: password; target: block; signalName: "passwordResult" }
    TestCase {
        name: "FacePassLockScreen"
        when: windowShown
        function test_faceButtonAndPasswordFallback() {
            var button = findChild(block, "facePassButton");
            verify(button !== null);
            compare(button.visible, false);
            authenticator.infoMessage = "FacePass is ready. Use your password.";
            tryCompare(button, "visible", true);
            compare(button.text, "FacePass");
            block.mainPasswordBox.text = "existing password";
            button.clicked();
            compare(password.count, 1);
            compare(password.signalArguments[0][0], "");
            compare(block.mainPasswordBox.text, "");
            compare(button.enabled, false);
            authenticator.failed(0);
            compare(button.enabled, true);
            block.mainPasswordBox.text = "normal-password";
            block.startLogin();
            compare(password.count, 2);
            compare(password.signalArguments[1][0], "normal-password");
        }
    }
}
'''


def main():
    source = Path("/usr/share/plasma/shells/org.kde.plasma.desktop/contents/lockscreen")
    with tempfile.TemporaryDirectory(prefix="facepass-qml-test-") as directory:
        target = Path(directory)
        shutil.copytree(source, target, dirs_exist_ok=True)
        (target / "MainBlock.qml").write_text(patch_main((target / "MainBlock.qml").read_text()))
        (target / "LockScreenUi.qml").write_text(patch_messages((target / "LockScreenUi.qml").read_text()))
        (target / "tst_facepass.qml").write_text(TEST)
        # This enum is registered by the greeter executable, not an installed
        # QML plugin. Supply only its two constants for this component test.
        registry = target / "imports/org/kde/kscreenlocker"
        registry.mkdir(parents=True)
        (registry / "qmldir").write_text("module org.kde.kscreenlocker\nsingleton Authenticator 1.0 Authenticator.qml\n")
        (registry / "Authenticator.qml").write_text("pragma Singleton\nimport QtQml\nQtObject { enum Kind { Fingerprint = 1, Smartcard = 2 } }\n")
        result = subprocess.run(["/usr/bin/qmltestrunner-qt6", "-import", str(target / "imports"), "-input", str(target / "tst_facepass.qml")],
                                env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_QUICK_BACKEND": "software",
                                     "QT_LOGGING_TO_CONSOLE": "1", "QT_FORCE_STDERR_LOGGING": "1"},
                                capture_output=True, text=True, timeout=30)
        print(json.dumps({"kde_component_check_passed": result.returncode == 0,
                          "output": (result.stdout + result.stderr)[-5000:]}))
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
