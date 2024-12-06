"""
select-test-imgs.py
---------------------
This is the script that was run to randomly select 10% of our original
dataset for use in testing the performance of ChatGPT. After running
`python3 select-test-imgs.py`, 83 images were randomly selected and copied over
to the test folder.
"""
import os
import shutil
import random

def copy_random_images(src_dir, dest_dir, percentage=10):
    # create 'test' folder if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # collect all imgs from /img - all our images have .jpeg
    all_images = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpeg':
                all_images.append(os.path.join(root, file))
    
    num_to_select = max(1, int(len(all_images) * (percentage / 100)))
    selected_images = random.sample(all_images, num_to_select)
    for image_path in selected_images:
        shutil.copy(image_path, dest_dir)
    
    print(f"Copied {len(selected_images)} images to '{dest_dir}'.")

source_directory = os.getcwd()
destination_directory = os.path.join(source_directory, 'test')
copy_random_images(source_directory, destination_directory)
