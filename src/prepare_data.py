import os
import shutil
import random
import matplotlib.pyplot as plt
import glob

# --- CONFIGURARE AUTOMATĂ ---
# Aflăm unde se află acest script (în folderul src)
script_location = os.path.dirname(os.path.abspath(__file__))
# Mergem un nivel mai sus, în folderul principal al proiectului
project_root = os.path.dirname(script_location)

# Definim căile exacte către date
raw_dir = os.path.join(project_root, "data", "raw")
base_dir = os.path.join(project_root, "data")
output_dirs = ["train", "validation", "test"]
split_ratios = (0.7, 0.15, 0.15) 

# Verificare de siguranță
if not os.path.exists(raw_dir):
    print(f"EROARE CRITICĂ: Nu găsesc folderul cu poze la calea:\n{raw_dir}")
    print("Verifică dacă ai pus folderele 'glass', 'paper' etc. în data/raw!")
    exit()

# 1. Funcție pentru împărțirea datelor
def split_dataset():
    print(f"Căutăm date în: {raw_dir}")
    print("Începem împărțirea datelor...")
    
    # Găsim clasele (folderele)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if not classes:
        print("Nu am găsit niciun folder de clasă (glass, paper etc.)!")
        return []

    # Creăm folderele train, validation, test
    for split in output_dirs:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    # Procesăm fiecare clasă
    total_images = 0
    for cls in classes:
        src_path = os.path.join(raw_dir, cls)
        images = os.listdir(src_path)
        
        # Filtrăm doar fișierele (imagini), ignorăm folderele ascunse
        images = [img for img in images if os.path.isfile(os.path.join(src_path, img))]
        
        random.shuffle(images)
        
        train_end = int(len(images) * split_ratios[0])
        val_end = train_end + int(len(images) * split_ratios[1])
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        def copy_files(file_list, destination):
            for file_name in file_list:
                shutil.copy(
                    os.path.join(src_path, file_name),
                    os.path.join(base_dir, destination, cls, file_name)
                )

        copy_files(train_imgs, "train")
        copy_files(val_imgs, "validation")
        copy_files(test_imgs, "test")
        
        count = len(images)
        total_images += count
        print(f"Clasa '{cls}': {count} imagini procesate.")

    print(f"\nSucces! Total imagini procesate: {total_images}")
    return classes

# 2. Funcție pentru generarea graficului (EDA)
def generate_eda_chart(classes):
    if not classes:
        return

    print("\nGenerăm graficul de distribuție...")
    counts = []
    for cls in classes:
        count = len(os.listdir(os.path.join(raw_dir, cls)))
        counts.append(count)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=['#CD853F', '#708090', '#C0C0C0', '#F5F5DC', '#87CEEB', '#2F4F4F'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

    plt.title("Distribuția Claselor în Setul de Date")
    plt.xlabel("Tip Deșeu")
    plt.ylabel("Număr Imagini")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Salvăm în folderul docs/images
    save_dir = os.path.join(project_root, "docs", "images")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "distributie_clase.png")
    
    plt.savefig(save_path)
    print(f"Graficul a fost salvat în: {save_path}")

if __name__ == "__main__":
    lista_clase = split_dataset()
    generate_eda_chart(lista_clase)