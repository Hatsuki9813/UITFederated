import os

# Thư mục gốc của dự án
root_dir = "flower_dashboard"

# Danh sách các thư mục cần tạo
directories = [
    os.path.join(root_dir, "app", "pages"),
    os.path.join(root_dir, "app", "components"),
    os.path.join(root_dir, "app", "utils"),
    os.path.join(root_dir, "config"),
    os.path.join(root_dir, "data"),
    os.path.join(root_dir, "models"),
    os.path.join(root_dir, "flower_logic")
]

# Danh sách các file cần tạo (đường dẫn tương đối từ thư mục gốc)
files = [
    os.path.join(root_dir, "app", "__init__.py"),
    os.path.join(root_dir, "app", "main.py"),
    os.path.join(root_dir, "app", "pages", "__init__.py"),
    os.path.join(root_dir, "app", "pages", "dashboard.py"),
    os.path.join(root_dir, "app", "pages", "training.py"),
    os.path.join(root_dir, "app", "pages", "admin.py"),
    os.path.join(root_dir, "app", "pages", "settings.py"),
    os.path.join(root_dir, "app", "components", "__init__.py"),
    os.path.join(root_dir, "app", "utils", "__init__.py"),
    os.path.join(root_dir, "config", "config.yaml"),
    os.path.join(root_dir, "data", ".gitkeep"),
    os.path.join(root_dir, "models", ".gitkeep"),
    os.path.join(root_dir, "flower_logic", "__init__.py"),
    os.path.join(root_dir, "flower_logic", "client.py"),
    os.path.join(root_dir, "flower_logic", "server.py"),
    os.path.join(root_dir, "flower_logic", "utils.py"),
    os.path.join(root_dir, "requirements.txt"),
    os.path.join(root_dir, "README.md"),
    os.path.join(root_dir, ".gitignore")
]

# Tạo thư mục gốc nếu nó chưa tồn tại
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
    print(f"Đã tạo thư mục gốc: {root_dir}")
else:
    print(f"Thư mục gốc '{root_dir}' đã tồn tại.")

# Tạo các thư mục con
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Đã tạo thư mục: {directory}")
    else:
        print(f"Thư mục '{directory}' đã tồn tại.")

# Tạo các file
for file_path in files:
    if not os.path.exists(file_path):
        # Đảm bảo thư mục cha tồn tại trước khi tạo file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            # Tạo file rỗng
            pass
        print(f"Đã tạo file: {file_path}")
    else:
        print(f"File '{file_path}' đã tồn tại.")

print("Đã tạo xong cấu trúc thư mục và các file cơ bản.")