# Student: Magdalena Czapiewska

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog, daisy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score

SIZE = 64
INPUT_FILE = "input.txt"
OUTPUT_HTML = "output_clusters_images.html"
OUTPUT_TXT = "output_clusters.txt"
THRESHOLD = 1.4
GROUND_TRUTH = "ground_truth_clusters.txt"
FEATURES_EXTRACTION_ALGORITHM = "daisy" # possible values: hog, daisy
LINKAGE_METHOD = "ward" # possible values: ward, single, average, complete

def process_image(file_path, size, debug=False):
    """
    Loads, preprocesses, and normalizes a character image.
    
    Args:
        file_path (str): Path to the image file.
        size (int): Target size (width and height) of the output image.
        debug (bool): If True, displays the original and processed image with matrix values.
        
    Returns:
        numpy.ndarray: Normalized 2D image of shape (size, size) with float32 values, 
                       or None if loading fails.
    """
    # img is a 2-dimensional numpy.ndarray with entities of type uint8
    # each pixel is represented by one entity
    # values range from 0 (black) to 255 (white)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load file {file_path}")
        return None

    # The goal of the following operations is to find a bounding box around the letter in the image
    # values are inversed so that the letter is white (255) and the backgound is black (0)
    # Otsu's algorithm is used for analysing histogram of pixel values and choosing a threshold value
    # that separates letter and background
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # coordinates of non-zero pixels (white, letter) are found 
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.ones((size, size), dtype=np.float32)

    # the smallest rectangle (x, y, width, height) that contains all non-zero pixels is calculated
    x, y, w, h = cv2.boundingRect(coords)
    # the original grayscale image is cropped to this rectangle to remove empty margins
    # x, y: coordinates of the top-left corner of the bounding box
    # w, h: width and height of the box
    # slicing in NumPy [y:y+h, x:x+w] extracts the region of interest
    # by selecting rows (vertical range) and columns (horizontal range)
    # cropped_original contains a letter (0, black, as original) and fragments of background (255, white, original)
    cropped_original = img[y:y+h, x:x+w]

    # To avoid distorting the character, its original aspect ratio is preserved.
    # The scaling factor is calculated based on the larger dimension (width or height),
    # ensuring the resized character fits within a SIZE x SIZE box
    scaling_factor = float(size) / max(h, w)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)

    # Fragment of the original image (center with a letter, without margins) is resized
    # so that the larger dimension reaches SIZE and proportions are preserved.
    # INTER_AREA algorithm overlays a grid of the target resolution onto the source image.
    # It calculates the values of a new pixel by looking at the overlap with the original pixels
    # and computing the weighted average of all pixels covered by the target pixel's area.
    resized = cv2.resize(cropped_original, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # canvas is white (255)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    # The top-left corner coordinates (start_x, start_y) are calculated
    # to place the character in the center of the canvas.
    start_x = (size - new_w) // 2
    start_y = (size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # values are normalized to 0.0 (black) - 1.0 (white) range (float32)
    final_img_norm = canvas.astype(np.float32) / 255.0

    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Original")
        
        ax[1].imshow(final_img_norm, cmap='gray', vmin=0, vmax=1)
        ax[1].set_title(f"Normalized {size}x{size} (0-1)")
        plt.show()
        
        start_idx = size // 2 - 5
        end_idx = start_idx + 10
        
        print(f"\n### MATRIX VALUES (10x10 center fragment from index {start_idx} to {end_idx}) ###")
        print(final_img_norm[start_idx:end_idx, start_idx:end_idx])

    return final_img_norm

def load_image_paths(file_path):
    """
    Reads image file paths from a text file.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        list: A list of cleaned strings, each representing a path to an image.
    """
    image_paths = []

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # .strip() removes leading/trailing whitespaces and newline characters (\n)
            path = line.strip()
            if path:
                image_paths.append(path)
                
    return image_paths

def generate_reports(clusters_dict, output_html, output_txt):
    """
    Generates HTML and TXT reports based on clustering results.
    
    Args:
        clusters_dict (dict): Dictionary where keys are cluster IDs and values are lists of image paths.
        output_html (str): Filename for the HTML output.
        output_txt (str): Filename for the TXT output.
    """
    active_cluster_ids = sorted([cid for cid, paths in clusters_dict.items() if paths])

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        
        for i, cid in enumerate(active_cluster_ids):
            for path in clusters_dict[cid]:
                filename = os.path.basename(path)
                f.write(f'<img src="{path}" title="{filename}" style="margin:2px; max-height:50px;">\n')
            
            if i < len(active_cluster_ids) - 1:
                f.write("<HR>\n")
                
        f.write("</body></html>\n")

    with open(output_txt, "w", encoding="utf-8", newline="\n") as f:
        for cid in active_cluster_ids:
            filenames = [os.path.basename(p) for p in clusters_dict[cid]]
            f.write(" ".join(filenames) + "\n")

def process_and_vectorize_images(paths, size):
    """
    Iterates through a list of image paths, processes each image, and flattens it.
    
    Args:
        paths (list): List of image file paths.
        size (int): The target size for resizing images.
        
    Returns:
        tuple: (X, valid_paths) where:
            - X (numpy.ndarray): Feature matrix of shape (n_samples, size*size).
            - valid_paths (list): List of paths corresponding to the rows in X.
    """
    processed_images = []
    valid_paths = []

    for p in paths:
        proc_img = process_image(p, size, debug=False)
        
        if proc_img is not None:
            processed_images.append(proc_img.flatten())
            valid_paths.append(p)

    X = np.array(processed_images)
    return X, valid_paths

def extract_hog_features(X_images, size, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from a set of images.
    
    Args:
        X_images (numpy.ndarray): Matrix of flattened images.
        size (int): The original side length of the square images (size x size).
        orientations (int): Number of orientation bins.
        pixels_per_cell (tuple): Size (in pixels) of a cell.
        cells_per_block (tuple): Number of cells in each block.
        
    Returns:
        numpy.ndarray: A 2D array where each row is a HOG feature vector for an image.
    """
    hog_features = []
    
    for img_flat in X_images:
        img = img_flat.reshape(size, size)
        
        features = hog(img, 
                       orientations=orientations, 
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, 
                       visualize=False)
        
        hog_features.append(features)
        
    return np.array(hog_features)

def extract_daisy_features(X_images, size, step=16, radius=15, rings=2, histograms=6, orientations=8):
    """
    Extracts DAISY features from a set of images.
    
    Args:
        X_images (numpy.ndarray): Matrix of flattened images.
        size (int): The original side length of the square images (size x size).
        step (int): Distance between descriptor sampling points.
        radius (int): Radius of the outermost ring.
        rings (int): Number of rings in the descriptor grid.
        histograms (int): Number of histograms sampled per ring.
        orientations (int): Number of orientations in each histogram.
        
    Returns:
        numpy.ndarray: A 2D array of flattened DAISY descriptors for each image.
    """
    daisy_features = []
    
    for img_flat in X_images:
        img = img_flat.reshape(size, size)

        features = daisy(img, 
                         step=step, 
                         radius=radius, 
                         rings=rings, 
                         histograms=histograms, 
                         orientations=orientations, 
                         visualize=False)

        daisy_features.append(features.flatten())
        
    return np.array(daisy_features)

def perform_clustering(X_features, valid_paths, n_clusters=None, distance_threshold=0.5, metric='euclidean', linkage='ward'):
    """
    Performs hierarchical agglomerative clustering on the extracted features.
    
    Args:
        X_features (numpy.ndarray): Feature matrix (n_samples, n_features).
        valid_paths (list): List of image paths corresponding to the rows in X_features.
        n_clusters (int, optional): The number of clusters to find. Should be None if 
                                     distance_threshold is used.
        distance_threshold (float): The linkage distance threshold above which 
                                    clusters will not be merged.
        metric (str): Metric used to compute the linkage.
        linkage (str): Linkage criterion.
        
    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of 
              image paths belonging to each cluster.
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage=linkage
    )

    labels = model.fit_predict(X_features)

    clusters_dict = {}
    for path, label in zip(valid_paths, labels):
        clusters_dict.setdefault(label, []).append(path)
    
    return clusters_dict

def calculate_clustering_metrics(file_true, file_pred):
    """
    Calculates Rand Index (RI) and Adjusted Rand Index (ARI) by comparing 
    the predicted clustering with the ground truth.
    
    Args:
        file_true (str): Path to the ground truth TXT file.
        file_pred (str): Path to the txt file generated by the program.
        
    Returns:
        tuple: (ri, ari) scores.
    """
    def parse_to_labels(file_path):
        name_to_label = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for cluster_id, line in enumerate(f):
                filenames = line.strip().split()
                for name in filenames:
                    name_to_label[name] = cluster_id
        return name_to_label

    labels_true_dict = parse_to_labels(file_true)
    labels_pred_dict = parse_to_labels(file_pred)

    common_names = sorted(list(set(labels_true_dict.keys()) & set(labels_pred_dict.keys())))
    
    if not common_names:
        print("Error: No common filenames found between the two files.")
        return 0.0, 0.0

    y_true = [labels_true_dict[name] for name in common_names]
    y_pred = [labels_pred_dict[name] for name in common_names]

    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return ri, ari


if __name__ == "__main__":
    all_paths = load_image_paths(INPUT_FILE)
    if not all_paths:
        print("Exiting: No input images found.")
        exit()
    X, valid_paths = process_and_vectorize_images(all_paths, SIZE)

    print(f"Total paths read from {INPUT_FILE}: {len(all_paths)}")
    print(f"Successfully processed images:    {len(valid_paths)}")

    if X.size > 0:
        print(f"Matrix X shape (samples, features): {X.shape}")
        print(f"Feature vector size per image:      {X.shape[1]} (equal to {SIZE}x{SIZE})")
    else:
        print("Warning: Matrix X is empty. Check your image paths or process_image function.")

    if FEATURES_EXTRACTION_ALGORITHM.lower() == "daisy":
        print(f"Extracting features using DAISY algorithm...")
        X_features = extract_daisy_features(X, SIZE)
    
    elif FEATURES_EXTRACTION_ALGORITHM.lower() == "hog":
        print(f"Extracting features using HOG algorithm...")
        X_features = extract_hog_features(X, SIZE)
      
    else:
        print("Warning: Unknown algorithm selected. Using raw pixel values.")
        X_features = X

    if X_features.size > 0:
        print("-" * 30)
        print(f"### FEATURE EXTRACTION SUMMARY ###")
        print(f"Algorithm used:           {FEATURES_EXTRACTION_ALGORITHM.upper()}")
        print(f"Feature Matrix shape:     {X_features.shape}")
        print(f"Number of samples:        {X_features.shape[0]}")
        print(f"Features per image:       {X_features.shape[1]}")
        print("-" * 30)
    else:
        print(f"Warning: Feature matrix is empty! Check the {FEATURES_EXTRACTION_ALGORITHM} settings.")

    clusters = perform_clustering(
        X_features, 
        valid_paths, 
        n_clusters=None, 
        distance_threshold=THRESHOLD, 
        metric='euclidean',
        linkage=LINKAGE_METHOD
    )

    generate_reports(clusters, OUTPUT_HTML, OUTPUT_TXT)

    print("-" * 30)
    print("### FINAL REPORT SUMMARY ###")
    print(f"Features extraction algorithm used:    {FEATURES_EXTRACTION_ALGORITHM.upper()}")
    print(f"Linkage method:    {LINKAGE_METHOD}")
    print(f"Threshold:         {THRESHOLD}")
    print(f"Clusters found:    {len(clusters)}")
    print("-" * 30)
    print(f"Results successfully saved to:")
    print(f" - {OUTPUT_HTML}")
    print(f" - {OUTPUT_TXT}")
    print("-" * 30)
  
    #if os.path.exists(GROUND_TRUTH):
    #    ri, ari = calculate_clustering_metrics(GROUND_TRUTH, OUTPUT_TXT)
    #    print(f"Rand Index: {ri:.4f}")
    #    print(f"Adjusted Rand Index: {ari:.4f}")
    #else:
    #    print(f"Ground truth file '{GROUND_TRUTH}' not found. Skipping evaluation.")
