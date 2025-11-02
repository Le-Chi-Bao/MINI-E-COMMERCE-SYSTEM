import json
import hashlib
import os
from typing import Dict, Optional
from datetime import datetime

USER_DATA_FILE = "info_login.json"

class SimpleAuth:
    def __init__(self, user_file=USER_DATA_FILE):
        self.user_file = user_file
        self.users = self.load_users()
    
    def load_users(self) -> Dict:
        """Load users từ JSON file với UTF-8 encoding"""
        if os.path.exists(self.user_file):
            try:
                with open(self.user_file, 'r', encoding='utf-8') as f:
                    data = f.read().strip()
                    if not data:  # File rỗng
                        return self._create_default_users()
                    return json.loads(data)
            except json.JSONDecodeError:
                print("⚠️ File info_login.json bị lỗi, tạo file mới...")
                return self._create_default_users()
            except Exception as e:
                print(f"❌ Lỗi khi load users: {e}")
                return self._create_default_users()
        
        # File không tồn tại, tạo user mặc định
        return self._create_default_users()
    
    def _create_default_users(self) -> Dict:
        """Tạo users mặc định"""
        default_users = {
            "admin": {
                "password": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
                "email": "admin@system.com",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "admin"
            }
        }
        self._save_users_safe(default_users)
        return default_users
    
    def _save_users_safe(self, users_data: Dict):
        """Lưu users an toàn với encoding UTF-8"""
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(self.user_file) if os.path.dirname(self.user_file) else '.', exist_ok=True)
            
            # Lưu với encoding UTF-8 và ensure_ascii=False
            with open(self.user_file, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Đã lưu {len(users_data)} users vào {self.user_file}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu users: {e}")
            # Thử cách khác nếu cách trên lỗi
            try:
                with open(self.user_file, 'w', encoding='utf-8') as f:
                    # Dùng ensure_ascii=True để tránh lỗi encoding
                    json.dump(users_data, f, indent=4, ensure_ascii=True)
                print(f"✅ Đã lưu users (dùng ensure_ascii=True)")
            except Exception as e2:
                print(f"❌ Lỗi nghiêm trọng khi lưu users: {e2}")
    
    def save_users(self):
        """Lưu users vào JSON file"""
        self._save_users_safe(self.users)
    
    def hash_password(self, password: str) -> str:
        """Hash password đơn giản"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def register(self, username: str, password: str, email: str = "") -> Dict:
        """Đăng ký user mới"""
        try:
            # Validate input
            if not username or not password:
                return {"success": False, "message": "Username và password không được để trống!"}
            
            if username in self.users:
                return {"success": False, "message": "Username đã tồn tại!"}
            
            if len(username) < 3:
                return {"success": False, "message": "Username phải có ít nhất 3 ký tự!"}
            
            if len(password) < 4:
                return {"success": False, "message": "Password phải có ít nhất 4 ký tự!"}
            
            # Tạo user mới
            self.users[username] = {
                "password": self.hash_password(password),
                "email": email,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "user"
            }
            
            # Lưu vào file
            self.save_users()
            return {"success": True, "message": "Đăng ký thành công!"}
            
        except Exception as e:
            return {"success": False, "message": f"Lỗi hệ thống: {str(e)}"}
    
    def login(self, username: str, password: str) -> Dict:
        """Đăng nhập"""
        try:
            if not username or not password:
                return {"success": False, "message": "Username và password không được để trống!"}
            
            if username in self.users:
                stored_hash = self.users[username]["password"]
                if stored_hash == self.hash_password(password):
                    return {
                        "success": True, 
                        "message": "Đăng nhập thành công!",
                        "user_info": {
                            "username": username,
                            "email": self.users[username].get("email", ""),
                            "role": self.users[username].get("role", "user")
                        }
                    }
            
            return {"success": False, "message": "Sai tên đăng nhập hoặc mật khẩu!"}
            
        except Exception as e:
            return {"success": False, "message": f"Lỗi hệ thống: {str(e)}"}
    
    def get_all_users(self) -> Dict:
        """Lấy danh sách users (cho admin)"""
        return self.users

# Khởi tạo auth system
auth_system = SimpleAuth()