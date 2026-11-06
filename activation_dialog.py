"""
Modern PyQt6 License Activation & Renewal Dialog for Cambo Dubber (Csophouret)
Provides a secure locked screen preventing access until a valid, unexpired license is entered.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QFrame, QApplication, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

from icons import get_google_icon
from license_manager import LicenseManager


class ActivationDialog(QDialog):
    def __init__(self, license_manager: LicenseManager = None, status_info: dict = None, parent=None):
        super().__init__(parent)
        self.mgr = license_manager or LicenseManager()
        self.status_info = status_info or self.mgr.check_license()

        self.setWindowTitle("ប្រព័ន្ធផ្ទៀងផ្ទាត់អាជ្ញាបណ្ណ - Csophouret (Cambo Dubber)")
        self.setFixedSize(580, 620)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)

        self.init_ui()

    def init_ui(self):
        # Universal Dialog Stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Segoe UI', 'Kantumruy Pro', 'Khmer OS Battambang', sans-serif;
            }
            QLabel {
                color: #f1f5f9;
            }
            QLineEdit, QTextEdit {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                color: #f8fafc;
                font-size: 13px;
                selection-background-color: #0284c7;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #38bdf8;
                background-color: #111c35;
            }
            QPushButton {
                font-family: 'Segoe UI', 'Kantumruy Pro', sans-serif;
                font-size: 13px;
                font-weight: 600;
                border-radius: 6px;
                padding: 9px 18px;
            }
            QPushButton.btn-primary {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #0369a1;
            }
            QPushButton.btn-primary:hover {
                background-color: #0ea5e9;
                border-color: #38bdf8;
            }
            QPushButton.btn-primary:pressed {
                background-color: #0369a1;
            }
            QPushButton.btn-secondary {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #475569;
            }
            QPushButton.btn-secondary:hover {
                background-color: #334155;
                color: #ffffff;
                border-color: #64748b;
            }
            QPushButton.btn-copy {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #0284c7;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton.btn-copy:hover {
                background-color: #0284c7;
                color: #ffffff;
            }
            QFrame.card {
                background-color: #131c31;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame.contact-card {
                background-color: #0e2038;
                border: 1px solid #0284c7;
                border-radius: 8px;
            }
            /* Right-Click Context Menu */
            QMenu {
                background-color: #0f172a;
                color: #38bdf8;
                border: 1px solid #0284c7;
                border-radius: 6px;
                padding: 5px;
                font-size: 12px;
                font-weight: 500;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 24px 6px 14px;
                border-radius: 4px;
                color: #38bdf8;
            }
            QMenu::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #64748b;
                background-color: transparent;
            }
            QMenu::separator {
                height: 1px;
                background-color: #334155;
                margin: 4px 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 1. Header (Icon + Title)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(get_google_icon("shield", color="#38bdf8", size=40).pixmap(40, 40))
        header_layout.addWidget(icon_lbl)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        app_title = QLabel("Csophouret Dubbing Suite")
        app_title.setStyleSheet("font-size: 17px; font-weight: bold; color: #38bdf8;")
        title_col.addWidget(app_title)

        sub_title = QLabel("ប្រព័ន្ធគ្រប់គ្រង និងផ្ទៀងផ្ទាត់អាជ្ញាបណ្ណ (License Activation)")
        sub_title.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_col.addWidget(sub_title)

        header_layout.addLayout(title_col)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # 2. Status Banner (Active, Expired, or Not Activated)
        status = self.status_info.get("status", "not_activated")
        self.banner = QFrame()
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(12, 10, 12, 10)
        banner_layout.setSpacing(10)

        self.banner_icon = QLabel()
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        self.banner_text.setStyleSheet("font-size: 12px; line-height: 1.4;")

        if status == "expired":
            self.banner.setStyleSheet("background-color: #450a0a; border: 1px solid #dc2626; border-radius: 6px;")
            self.banner_icon.setPixmap(get_google_icon("lock", color="#ef4444", size=24).pixmap(24, 24))
            exp_date = self.status_info.get("exp", "N/A")
            self.banner_text.setText(
                f"<b>⚠ អាជ្ញាបណ្ណរបស់អ្នកបានផុតកំណត់កាលពីថ្ងៃទី {exp_date}!</b><br>"
                "សូមទំនាក់ទំនង Admin តាមព័ត៌មានខាងក្រោមដើម្បីបន្តសុពលភាពថ្មី។"
            )
        elif status == "active":
            self.banner.setStyleSheet("background-color: #064e3b; border: 1px solid #10b981; border-radius: 6px;")
            self.banner_icon.setPixmap(get_google_icon("check_circle", color="#10b981", size=24).pixmap(24, 24))
            rem = self.status_info.get("remaining_days", 0)
            exp_date = self.status_info.get("exp", "LIFETIME")
            rem_str = "គ្មានដែនកំណត់ (Lifetime)" if self.status_info.get("is_lifetime") else f"{rem} ថ្ងៃទៀត (ផុតកំណត់: {exp_date})"
            self.banner_text.setText(
                f"<b>✓ អាជ្ញាបណ្ណកំពុងដំណើរការ (Active)</b><br>"
                f"សុពលភាពនៅសល់: {rem_str}"
            )
        else:
            self.banner.setStyleSheet("background-color: #2e1065; border: 1px solid #7c3aed; border-radius: 6px;")
            self.banner_icon.setPixmap(get_google_icon("key", color="#a78bfa", size=24).pixmap(24, 24))
            self.banner_text.setText(
                "<b>🔒 កម្មវិធីមិនទាន់បាន Activate នៅឡើយទេ</b><br>"
                "សូមចម្លងលេខកូដម៉ាស៊ីន (Machine ID) ផ្ញើមកកាន់ Admin តាមរយៈព័ត៌មានខាងក្រោម។"
            )

        banner_layout.addWidget(self.banner_icon)
        banner_layout.addWidget(self.banner_text, 1)
        layout.addWidget(self.banner)

        # 3. Prominent Admin Contact Card (Csophouret : 088 77 32326)
        contact_card = QFrame()
        contact_card.setProperty("class", "contact-card")
        contact_layout = QHBoxLayout(contact_card)
        contact_layout.setContentsMargins(12, 10, 12, 10)
        contact_layout.setSpacing(12)

        c_icon = QLabel()
        c_icon.setPixmap(get_google_icon("verified", color="#38bdf8", size=32).pixmap(32, 32))
        contact_layout.addWidget(c_icon)

        c_info = QVBoxLayout()
        c_info.setSpacing(2)

        c_title = QLabel("📞 ទំនាក់ទំនងទិញ ឬបើក License Key (Contact Admin):")
        c_title.setStyleSheet("font-size: 11.5px; color: #94a3b8; font-weight: bold;")
        c_info.addWidget(c_title)

        c_phone = QLabel("Csophouret :  088 77 32326")
        c_phone.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8; letter-spacing: 0.5px;")
        c_info.addWidget(c_phone)

        c_sub = QLabel("Telegram / Call — ផ្ញើ Machine ID មកកាន់លេខនេះដើម្បីទទួលបាន Key")
        c_sub.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        c_info.addWidget(c_sub)

        contact_layout.addLayout(c_info, 1)

        self.btn_copy_phone = QPushButton("📋 ចម្លងលេខ")
        self.btn_copy_phone.setProperty("class", "btn-copy")
        self.btn_copy_phone.setToolTip("ចុចដើម្បីចម្លងលេខទូរស័ព្ទ 088 77 32326")
        self.btn_copy_phone.clicked.connect(self.on_copy_phone)
        contact_layout.addWidget(self.btn_copy_phone)

        layout.addWidget(contact_card)

        # 4. Machine ID Section
        mid_card = QFrame()
        mid_card.setProperty("class", "card")
        mid_layout = QVBoxLayout(mid_card)
        mid_layout.setContentsMargins(12, 10, 12, 10)
        mid_layout.setSpacing(6)

        mid_header = QHBoxLayout()
        mid_lbl = QLabel("លេខកូដសម្គាល់ម៉ាស៊ីន (Machine ID):")
        mid_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #cbd5e1;")
        mid_header.addWidget(mid_lbl)
        mid_header.addStretch()

        self.copy_btn = QPushButton("📋 ចម្លងកូដ (Copy Machine ID)")
        self.copy_btn.setProperty("class", "btn-copy")
        self.copy_btn.clicked.connect(self.on_copy_machine_id)
        mid_header.addWidget(self.copy_btn)
        mid_layout.addLayout(mid_header)

        self.mid_edit = QLineEdit(self.mgr.machine_id)
        self.mid_edit.setReadOnly(True)
        self.mid_edit.setStyleSheet("""
            QLineEdit {
                background-color: #0b0f19;
                border: 1px dashed #475569;
                color: #38bdf8;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        mid_layout.addWidget(self.mid_edit)

        layout.addWidget(mid_card)

        # 4. License Key Input Section
        key_card = QFrame()
        key_card.setProperty("class", "card")
        key_layout = QVBoxLayout(key_card)
        key_layout.setContentsMargins(12, 10, 12, 10)
        key_layout.setSpacing(6)

        key_header = QHBoxLayout()
        key_lbl = QLabel("បញ្ចូល License Key របស់អ្នក:")
        key_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #cbd5e1;")
        key_header.addWidget(key_lbl)
        key_header.addStretch()

        paste_btn = QPushButton("📥 បិទភ្ជាប់ (Paste)")
        paste_btn.setProperty("class", "btn-copy")
        paste_btn.clicked.connect(self.on_paste_key)
        key_header.addWidget(paste_btn)
        key_layout.addLayout(key_header)

        self.key_edit = QTextEdit()
        self.key_edit.setPlaceholderText("Paste CDKEY-... នៅទីនេះ")
        self.key_edit.setFixedHeight(75)
        self.key_edit.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        key_layout.addWidget(self.key_edit)

        layout.addWidget(key_card)

        # 5. Message / Status Feedback Label
        self.feedback_lbl = QLabel("")
        self.feedback_lbl.setWordWrap(True)
        self.feedback_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_lbl.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(self.feedback_lbl)

        # 6. Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_exit = QPushButton("🚪 ចាកចេញ (Exit)")
        self.btn_exit.setProperty("class", "btn-secondary")
        self.btn_exit.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_exit)

        btn_layout.addStretch()

        self.btn_activate = QPushButton("🚀 ដំណើរការកម្មវិធី (Activate License)")
        self.btn_activate.setIcon(get_google_icon("verified", color="#ffffff"))
        self.btn_activate.setProperty("class", "btn-primary")
        self.btn_activate.clicked.connect(self.on_activate_clicked)
        btn_layout.addWidget(self.btn_activate)

        layout.addLayout(btn_layout)

    def on_copy_machine_id(self):
        """Copies the Machine ID to clipboard and provides feedback."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.mgr.machine_id)
        self.copy_btn.setText("✓ បានចម្លងរួចរាល់!")
        self.copy_btn.setStyleSheet("background-color: #059669; color: #ffffff; border: 1px solid #10b981;")
        QTimer.singleShot(2500, self.reset_copy_btn)

    def reset_copy_btn(self):
        self.copy_btn.setText("📋 ចម្លងកូដ (Copy Machine ID)")
        self.copy_btn.setStyleSheet("")

    def on_copy_phone(self):
        """Copies admin phone number to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText("088 77 32326")
        self.btn_copy_phone.setText("✓ បានចម្លង!")
        self.btn_copy_phone.setStyleSheet("background-color: #059669; color: #ffffff; border: 1px solid #10b981;")
        QTimer.singleShot(2000, self.reset_copy_phone_btn)

    def reset_copy_phone_btn(self):
        self.btn_copy_phone.setText("📋 ចម្លងលេខ")
        self.btn_copy_phone.setStyleSheet("")

    def on_paste_key(self):
        """Pastes text from clipboard into the License Key edit."""
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            self.key_edit.setPlainText(text)

    def on_activate_clicked(self):
        """Validates and activates the entered key."""
        key_str = self.key_edit.toPlainText().strip()
        if not key_str:
            self.feedback_lbl.setText("❌ សូមបញ្ចូល License Key ជាមុនសិន!")
            self.feedback_lbl.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")
            return

        res = self.mgr.activate(key_str)
        if res.get("valid"):
            rem = res.get("remaining_days", 0)
            exp_date = res.get("exp", "LIFETIME")
            rem_str = "គ្មានដែនកំណត់ (Lifetime)" if res.get("is_lifetime") else f"{rem} ថ្ងៃ (ផុតកំណត់: {exp_date})"
            self.feedback_lbl.setText(f"🎉 ជោគជ័យ! អាជ្ញាបណ្ណត្រូវបានដំណើរការ (សុពលភាព: {rem_str})")
            self.feedback_lbl.setStyleSheet("color: #10b981; font-size: 13px; font-weight: bold;")

            # Disable buttons to prevent double-click
            self.btn_activate.setEnabled(False)
            self.btn_exit.setEnabled(False)

            # Auto close and proceed after 1 second
            QTimer.singleShot(1100, self.accept)
        else:
            reason = res.get("reason", "License Key មិនត្រឹមត្រូវ!")
            self.feedback_lbl.setText(f"❌ {reason}")
            self.feedback_lbl.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")
