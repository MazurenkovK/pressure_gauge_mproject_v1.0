import os
import pandas as pd

# Каталог с файлами
folder = r'app/data_model/generated/test/images'

# Проходим по всем файлам в папке
for filename in os.listdir(folder):
    # Проверяем, есть ли в имени точка (кроме расширения)
    if '.' in filename[:-4]:  # [-4:] — это .jpg/.png
        new_name = filename.replace('.', '_')
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        # Переименовываем
        os.rename(old_path, new_path)
        # print(f"🔄�️ {filename} → {new_name}")
print(f"{folder} обновлён!")

# Путь к CSV
csv_path = r"app/data_model/generated/test/labels.csv"

# Читаем файл
df = pd.read_csv(csv_path)

# Меняем точку на подчеркивание в колонке filename
df['filename'] = df['filename'].apply(
    lambda s: s.replace('.', '_') if '.' in s[:-4] else s
)

# Сохраняем обратно
df.to_csv(csv_path, index=False)
print("CSV обновлён!")