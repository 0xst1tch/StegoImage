import cv2
import numpy as np
import argparse
from tqdm import tqdm 
from time import sleep
from textwrap import wrap

# Function to convert the encoded text into pixel data
def divider(encoded_text):
    # Initialize lists to store binary and integer representations of pixel data
    pixel_data_bin = [] 
    pixel_data_int = []  
    # Convert the length of the encoded text to a binary string and pad with zeros to reach 16 bits
    length_bin = format(len(encoded_text), 'b').zfill(16)
    # Split the binary length string into pairs of 2 bits
    length_bin_pairs = wrap(length_bin, 2)
    # Iterate through the length_bin_pairs and convert each pair to its corresponding integer representation    
    for i in length_bin_pairs:
        if i == "00":
            pixel_data_int.append(0)
        elif i == "01":
            pixel_data_int.append(1)
        elif i == "10":
            pixel_data_int.append(2)
        elif i == "11":
            pixel_data_int.append(3)
    # Iterate through the encoded text, convert each character to its 8-bit binary representation, and split it into pairs of 2 bits
    for i in encoded_text:
        pixel_data_bin += wrap(str(int(bin(i)[2:])).zfill(8), 2)
    # Iterate through the pixel_data_bin list and convert each binary pair to its corresponding integer representation
    for i in pixel_data_bin:
        if (i == "00"):
            pixel_data_int.append(0)
        elif (i == "01"):
            pixel_data_int.append(1)
        elif (i == "10"):
            pixel_data_int.append(2)
        elif (i == "11"):
            pixel_data_int.append(3)    

    return pixel_data_int
# Function to deobfuscate the gray value of a pixel and append the result to data_read list
def deobfuscator(gray_frame, t, data_read, image):
    gray_value = gray_frame[t[0],t[1]]
    if gray_value % 4 == 0:
        data_read.append(0)
    if gray_value % 4 == 1:
        data_read.append(1)
    if gray_value % 4 == 2:
        data_read.append(2)
    if gray_value % 4 == 3:
        data_read.append(3)
    return
# Function to obfuscate the pixel data into the input image
def obfuscator(frame, ROI, pixel_data_int, index):
    # Make a copy of the input frame to avoid modifying the original
    frame_aux = np.copy(frame)
    # Calculate the remaining length of the text to embed
    text_length = len(pixel_data_int) - index
    cont = 0
    # Iterate through the Region of Interest (ROI) coordinates
    for t in ROI:
        if cont >= text_length:
            break
        # Use tuple unpacking to improve readability
        y, x = t  
        color_rgb = frame_aux[y, x]
        blue_value, green_value, red_value = color_rgb  
        if(blue_value > 250):
            blue_value -= 4
        if(green_value > 250):
            green_value -= 4
        if(red_value > 250):
            red_value -= 4
        case_error = 0
        # Calculate the weighted sum of RGB values outside the loop to avoid recalculating it
        weighted_sum = 0.299 * red_value + 0.587 * green_value + 0.114 * blue_value
        # Modify the RGB values to embed the message
        while pixel_data_int[index] != round(weighted_sum) % 4:
            blue_value += 1
            green_value += 1
            red_value += 1
            # Increment the weighted sum accordingly
            weighted_sum += (0.299 + 0.587 + 0.114)  
            if(case_error > 4):
                blue_value += 1
                break
            case_error += 1

        index += 1
        frame_aux[y, x] = [blue_value, green_value, red_value]
        cont += 1

    return frame_aux
# Function to embed the data in the input image
def embed_data_in_image(image, pixel_data_int):
    # Convert the input image to grayscale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the less perceptible regions (ROI) in the image
    ROI = find_less_perceptible_regions(image)
    # Check if there are enough regions to store the pixel_data_int
    if(len(ROI) < len(pixel_data_int)):
        return None, None
    
    total_data_store = 0
    # If there are enough regions, embed the data into the image using the obfuscator function
    if len(ROI) > 0:
        image_new = obfuscator(image, ROI, pixel_data_int, total_data_store)
        total_data_store += len(ROI)

    return image_new, gray_frame
# Function to extract data from the input image
def extract_data_from_image(image, image_gray):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ROI = find_less_perceptible_regions(image_gray)
    cont = 0
    data_read = []

    if len(ROI) > 0:
        for t in ROI:
            deobfuscator(gray_frame, t, data_read, image)
            cont+=1
    return get_data(data_read)
# Function to find the less perceptible regions in the input frame
def find_less_perceptible_regions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 200, 220)
    threshold = 0.05
    strong_edges = np.argwhere(edges > threshold * np.max(edges))
    return strong_edges
# Function to reconstruct the message from the decoded data
def get_data(dec_data):
    first = True
    bin_data = []
    bin_str = ""
    message = ""
    # Convert the decimal values in dec_data to their corresponding binary representations
    for i in (dec_data):
        if i == 0:
            bin_data.append("00")
        if i == 1:
            bin_data.append("01")
        if i == 2:
            bin_data.append("10")
        if i == 3:
            bin_data.append("11")
    # If this is the first iteration, extract the message length from the binary data
    if first:
        message_length = int("".join(bin_data[:8]), 2)
        first = False
        # Remove the message length bits
        bin_data = bin_data[8:]  
    # Iterate through the binary data, combining every 4 bits into a single character
    for i in range(0, message_length * 4, 4):
        bin_str = "".join(bin_data[i:i + 4])
        if(bin_str == ''):
            break
        message += chr(int(bin_str, 2))

    return message
# Function to read the text from the input file
def read_file(text_file):
    lines = ""
    input = open(text_file)
    for line in input:
        lines += line
    return lines
# Function to read the input image
def read_image(input_image):
    #load video 
    image = cv2.imread(input_image)
    return image
# Function to encode the text into ASCII format
def ascii_encode(text):
    encoded_text = [ord(c) for c in text]
    return encoded_text
# Function to decode the ASCII-encoded text
def ascii_decode(encoded_text):
    text = ""
    decoded_text = [chr(c) for c in encoded_text]
    for c in decoded_text:
        text += c 
    return text
# Function to check if a string is a valid integer
def is_valid_integer(str_num):
    try:
        int(str_num)
        return True
    except ValueError:
        return False 

# Create arguments
parser = argparse.ArgumentParser(description="Steganography tool")
parser.add_argument("--mode", "-m", choices=['encode', 'decode'], help="Encode text into an image")
parser.add_argument("--input_image", "-i", type=str, help="Path to the input image")
parser.add_argument("--input_gray_image", "-ig", type=str, help="Path to the input gray image")
parser.add_argument("--output_image", "-o", type=str, help="Path to the output image")
parser.add_argument("--output_gray_image", "-og", type=str, help="Path to the output gray image")
parser.add_argument("--text_file", "-t", type=str, help="Path to the text file")
# Parse arguments
args = parser.parse_args()
input_image = args.input_image
input_gray_image = args.input_gray_image
output_image = args.output_image
output_gray_image = args.output_gray_image
text_file = args.text_file

print("""
   _____  __                       ____                             
  / ___/ / /_ ___   ____ _ ____   /  _/____ ___   ____ _ ____ _ ___ 
  \__ \ / __// _ \ / __ `// __ \  / / / __ `__ \ / __ `// __ `// _'\'
 ___/ // /_ /  __// /_/ // /_/ /_/ / / / / / / // /_/ // /_/ //  __/
/____/ \__/ \___/ \__, / \____//___//_/ /_/ /_/ \__,_/ \__, / \___/ 
                 /____/                               /____/        
                                """)

# Check the mode and perform the corresponding action
if (args.mode == "encode"):
    if(args.input_image is None or args.output_image is None or args.text_file is None or args.output_gray_image is None):
        print("You need to specify an input and output image and a text file for encoding")
        print("usage: image_stego.py [-h] [--mode {encode,decode}] [--input_image INPUT_IMAGE] [--output_image OUTPUT_IMAGE] [--output_gray_image OUTPUT_GRAY_IMAGE] [--text_file TEXT_FILE]")
    else:
        # Read text and encode it
        text = read_file(text_file)

        if (len(text) < 65535):
            encoded_text = ascii_encode(text)
            pixel_data_int = divider(encoded_text)
            decode_text = ascii_decode(encoded_text)

            image = read_image(input_image)
            image_with_data, gray_img = embed_data_in_image(image, pixel_data_int)
            if(image_with_data is None):
                print("You need to use a different image, try to use a more noisy image\n")
            else:
                # Save the resulting image
                cv2.imwrite(output_image, image_with_data)
                cv2.imwrite(output_gray_image, gray_img)
                for i in tqdm(range(0, 100),
                    desc = "Obfuscating the text in the image"): 
                    sleep(0.01) 
                print()
                print("The following files have been generated:")
                print(" ", output_image)
                print(" ", output_gray_image)
                print()
        else:
            print("Max length excedeed, max: 65535")
elif (args.mode == "decode"):
    if(args.input_image is None or args.input_gray_image is None):
        print("You need to specify an input image for decoding")
        print("usage: image_stego.py [-h] [--mode {encode,decode}] [--input_image INPUT_IMAGE] [--input_gray_image OUTPUT_GRAY_IMAGE] [--output_image OUTPUT_IMAGE] [--text_file TEXT_FILE]")
    else:  
        for i in tqdm(range(0, 100),
                desc = "Extracting text from the inserted image"): 
                sleep(0.01) 
        print()
        # Read the output image
        image = cv2.imread(input_image)
        image_gray = cv2.imread(input_gray_image)
        extracted_data = extract_data_from_image(image, image_gray)
        save_to_file = input("Do you want to save the decoded text to a file? (y/n): ")
        if save_to_file.lower() == "yes" or save_to_file.lower() == "y":
            output_file_name = input("Enter the output file name (with extension): ")
            with open(output_file_name, 'w') as output_file:
                output_file.write(extracted_data)
            print(f"Decoded text saved to {output_file_name}")
        elif(save_to_file.lower() == "no" or save_to_file.lower() == "n"):
            print("\nObtained text:\n")
            print(extracted_data)
        else:
            print("File not saved")
        
else:
    print("You need to specify a mode (encode/decode)")
    print("usage: image_stego.py [-h] [--mode {encode,decode}] [--input_image INPUT_IMAGE] [--output_image OUTPUT_IMAGE] [--text_file TEXT_FILE]")