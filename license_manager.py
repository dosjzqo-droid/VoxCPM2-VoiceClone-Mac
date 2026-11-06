"""
License & Activation Manager for Cambo Dubber (Csophouret)
Provides secure hardware-bound licensing with HMAC-SHA256 signature,
anti-tampering checks, and expiration validation.
"""

import os
import sys
import json
import uuid
import ctypes
import hashlib
import hmac
import base64
import platform
from datetime import datetime, date, timedelta

# Master secret salt known only to the developer (Cambo Dubber / Csophouret)
SECRET_SALT = "CAMBO_DUBBER_AUTH_SECRET_KEY_@CSOPHOURET_2026_SECURE_SALT#!"

def get_machine_id() -> str:
    """
    Generates a unique, deterministic Machine ID bound to the Windows PC hardware.
    Uses Windows MachineGuid and System Drive Volume Serial Number.
    """
    guid_val = ""
    vol_val = 0

    # 1. Try Windows MachineGuid from Registry
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            )
            guid_val, _ = winreg.QueryValueEx(key, "MachineGuid")
            winreg.CloseKey(key)
        except Exception:
            guid_val = ""

        # 2. Try System Drive Volume Serial
        try:
            vol = ctypes.c_ulong()
            ctypes.windll.kernel32.GetVolumeInformationW("C:\\", None, 0, ctypes.byref(vol), None, None, None, 0)
            vol_val = vol.value
        except Exception:
            vol_val = 0
    elif sys.platform == "darwin":
        try:
            cmd = "ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID"
            out = subprocess.check_output(cmd, shell=True, text=True)
            guid_val = out.split('"')[-2]
        except Exception:
            guid_val = ""

    # Fallback if registry or volume is missing
    fallback_node = uuid.getnode()
    hostname = platform.node()

    raw_seed = f"{guid_val}-{vol_val}-{fallback_node}-{hostname}"
    h = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest().upper()
    return f"CMDUB-{h[0:4]}-{h[4:8]}-{h[8:12]}-{h[12:16]}"


def get_license_file_path() -> str:
    """Returns the persistent file path where the license key is stored."""
    # Priority 1: Alongside application
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_lic = os.path.join(base_dir, "license.lic")

    # Priority 2: In APPDATA / macOS Application Support
    if sys.platform == "darwin":
        appdata_dir = os.path.expanduser("~/Library/Application Support/CamboDubber")
        os.makedirs(appdata_dir, exist_ok=True)
        appdata_lic = os.path.join(appdata_dir, "license.lic")
        if os.path.exists(local_lic):
            return local_lic
        if os.path.exists(appdata_lic):
            return appdata_lic
        return appdata_lic

    appdata = os.environ.get("APPDATA")
    if appdata:
        appdata_dir = os.path.join(appdata, "CamboDubber")
        os.makedirs(appdata_dir, exist_ok=True)
        appdata_lic = os.path.join(appdata_dir, "license.lic")
        # If local exists, use it; otherwise use appdata
        if os.path.exists(local_lic):
            return local_lic
        if os.path.exists(appdata_lic):
            return appdata_lic
        return local_lic

    return local_lic


class LicenseManager:
    """Handles licensing, verification, activation, and storage."""

    def __init__(self):
        self.machine_id = get_machine_id()
        self.license_path = get_license_file_path()

    def generate_key(
        self,
        machine_id: str,
        days: int = 30,
        client_name: str = "Client",
        tier: str = "PRO",
        is_lifetime: bool = False,
        custom_exp_date: str = None
    ) -> str:
        """
        Admin generation function:
        Creates a signed CDKEY token for a specific Machine ID and duration.
        """
        mid_clean = machine_id.strip().upper()
        today = date.today()

        if is_lifetime or days == -1 or days >= 99999:
            exp_str = "LIFETIME"
        elif custom_exp_date:
            exp_str = custom_exp_date.strip()
        else:
            exp_date = today + timedelta(days=max(0, days))
            exp_str = exp_date.strftime("%Y-%m-%d")

        payload = {
            "mid": mid_clean,
            "exp": exp_str,
            "iat": today.strftime("%Y-%m-%d"),
            "name": client_name.strip() or "Client",
            "tier": tier.strip() or "PRO"
        }

        json_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
        body_b64 = base64.urlsafe_b64encode(json_bytes).decode("utf-8").rstrip("=")
        sig = hmac.new(SECRET_SALT.encode("utf-8"), body_b64.encode("utf-8"), hashlib.sha256).hexdigest()[:16].upper()

        return f"CDKEY-{body_b64}-{sig}"

    def verify_key(self, key_str: str) -> dict:
        """
        Verifies if a license key is valid, authentic, matching the current machine, and not expired.
        """
        if not key_str or not isinstance(key_str, str):
            return {"valid": False, "reason": "គ្មាន License Key ទេ (No key provided)", "status": "missing"}

        key_str = key_str.strip()
        parts = key_str.split("-")
        if len(parts) < 3 or parts[0] != "CDKEY":
            return {"valid": False, "reason": "ទម្រង់ License Key មិនត្រឹមត្រូវ (Invalid key format)", "status": "invalid_format"}

        body_b64 = parts[1]
        sig = parts[2]

        # 1. Verify HMAC Signature
        expected_sig = hmac.new(SECRET_SALT.encode("utf-8"), body_b64.encode("utf-8"), hashlib.sha256).hexdigest()[:16].upper()
        if not hmac.compare_digest(sig, expected_sig):
            return {"valid": False, "reason": "ហត្ថលេខា License Key មិនត្រឹមត្រូវ ឬត្រូវបានកែប្រែ (Invalid/Tampered key signature)", "status": "invalid_sig"}

        # 2. Decode Payload
        try:
            # Fix base64 padding
            padding_needed = (4 - len(body_b64) % 4) % 4
            padded_b64 = body_b64 + ("=" * padding_needed)
            payload_json = base64.urlsafe_b64decode(padded_b64.encode("utf-8")).decode("utf-8")
            payload = json.loads(payload_json)
        except Exception as e:
            return {"valid": False, "reason": f"មិនអាចអានទិន្នន័យ License បានទេ ({str(e)})", "status": "corrupt_data"}

        key_mid = payload.get("mid", "").upper()
        exp_str = payload.get("exp", "")
        name = payload.get("name", "User")
        tier = payload.get("tier", "PRO")

        # 3. Match Hardware ID (supports ALL wildcard for developer master testing)
        if key_mid != "ALL" and key_mid != self.machine_id.upper():
            return {
                "valid": False,
                "reason": f"License Key នេះសម្រាប់តែម៉ាស៊ីន ({key_mid}) មិនមែនសម្រាប់ម៉ាស៊ីននេះទេ ({self.machine_id})!",
                "status": "machine_mismatch"
            }

        # 4. Check Expiration
        today = date.today()
        is_lifetime = (exp_str == "LIFETIME")
        remaining_days = 99999

        if not is_lifetime:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                return {"valid": False, "reason": "កាលបរិច្ឆេទផុតកំណត់មិនត្រឹមត្រូវ!", "status": "invalid_date"}

            if today > exp_date:
                days_over = (today - exp_date).days
                return {
                    "valid": False,
                    "reason": f"អាជ្ញាបណ្ណបានផុតកំណត់កាលពី {days_over} ថ្ងៃមុន ({exp_str})!",
                    "status": "expired",
                    "exp": exp_str,
                    "name": name,
                    "tier": tier,
                    "remaining_days": 0,
                    "is_lifetime": False
                }

            remaining_days = (exp_date - today).days

        return {
            "valid": True,
            "reason": "អាជ្ញាបណ្ណត្រឹមត្រូវ (Active)",
            "status": "active",
            "name": name,
            "tier": tier,
            "exp": exp_str,
            "remaining_days": remaining_days,
            "is_lifetime": is_lifetime
        }

    def load_saved_key(self) -> str:
        """Loads key from storage file."""
        for path in [self.license_path, os.path.join(os.environ.get("APPDATA", "."), "CamboDubber", "license.lic")]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = f.read().strip()
                        if data:
                            return data
                except Exception:
                    pass
        return ""

    def save_key(self, key_str: str) -> bool:
        """Saves key both locally and to %APPDATA%."""
        success = False
        paths = [self.license_path]
        appdata = os.environ.get("APPDATA")
        if appdata:
            appdata_dir = os.path.join(appdata, "CamboDubber")
            os.makedirs(appdata_dir, exist_ok=True)
            paths.append(os.path.join(appdata_dir, "license.lic"))

        for p in paths:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(key_str.strip())
                success = True
            except Exception:
                pass
        return success

    def remove_saved_key(self):
        """Deletes license file on revoke."""
        paths = [self.license_path]
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(os.path.join(appdata, "CamboDubber", "license.lic"))
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def check_license(self) -> dict:
        """
        Main entry point to check application license status on startup.
        Loads existing key and verifies it.
        """
        # Automatic Lifetime Bypass for Admin / Developer PC
        if self.machine_id.upper() in [
            "CMDUB-0B87-E400-87F8-78BB",  # Csophouret (Developer Master PC)
        ]:
            return {
                "valid": True,
                "reason": "Admin Developer Master PC (Lifetime)",
                "status": "active",
                "name": "Admin (Csophouret)",
                "tier": "ADMIN MASTER",
                "exp": "LIFETIME",
                "remaining_days": 99999,
                "is_lifetime": True,
                "machine_id": self.machine_id
            }

        saved_key = self.load_saved_key()
        if not saved_key:
            return {
                "valid": False,
                "reason": "មិនទាន់មាន License Key នៅឡើយទេ (Please activate your license)",
                "status": "not_activated",
                "machine_id": self.machine_id,
                "remaining_days": 0
            }

        res = self.verify_key(saved_key)
        res["machine_id"] = self.machine_id
        res["key"] = saved_key
        return res

    def activate(self, key_str: str) -> dict:
        """
        Attempts to activate a new key. If valid, persists it.
        """
        res = self.verify_key(key_str)
        if res.get("valid"):
            self.save_key(key_str)
        return res
