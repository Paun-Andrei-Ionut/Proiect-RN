import os
import shutil
import random
import glob

# --- CONFIGURARE ---
script_location = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_location))

raw_dir = os.path.join(project_root, "data", "raw")
generated_dir = os.path.join(project_root, "data", "generated") # AICI E CHEIA
base_dir = os.path.join(project_root, "data")

output_dirs = ["train", "validation", "test"]
split_ratios = (0.7, 0.15, 0.15)
classes = ['glass', 'metal', 'paper', 'plastic']

def prepare_dataset():
    print(f"🔄 Recombinăm TOATE datele (Public + Sintetic)...")
    
    # Curățăm tot
    for d in output_dirs:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for c in classes:
            os.makedirs(os.path.join(path, c))

    total = 0
    for cls in classes:
        # 1. Luăm din RAW
        raw_path = os.path.join(raw_dir, cls)
        imgs_raw = glob.glob(os.path.join(raw_path, "*.*"))
        
        # 2. Luăm din GENERATED (Datele tale originale)
        gen_path = os.path.join(generated_dir, cls)
        imgs_gen = glob.glob(os.path.join(gen_path, "*.*"))

        # Le unim
        all_imgs = imgs_raw + imgs_gen
        # Păstrăm doar imagini valide
        all_imgs = [x for x in all_imgs if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

        random.shuffle(all_imgs)
        count = len(all_imgs)
        total += count
        
        train_end = int(count * 0.7)
        val_end = train_end + int(count * 0.15)
        
        for i, img in enumerate(all_imgs):
            fname = os.path.basename(img)
            # Dacă e duplicat numele, îl prefixăm
            if i < train_end: dest = "train"
            elif i < val_end: dest = "validation"
            else: dest = "test"
            
            try:
                shutil.copy(img, os.path.join(base_dir, dest, cls, f"{i}_{fname}"))
            except:
                pass

        print(f"✅ {cls}: {len(imgs_raw)} publice + {len(imgs_gen)} generate = {count} total")

    print(f"🎉 Total final: {total} imagini. Gata de antrenare!")

if __name__ == "__main__":
    prepare_dataset()