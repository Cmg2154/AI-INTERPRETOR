import zipfile

zip_path = r"C:\Users\User\OneDrive\Desktop\AI_Interpreter\asl-alphabet.zip"
extract_path = r"C:\Users\User\OneDrive\Desktop\AI_Interpreter\data\asl_alphabet"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ 解压完成")