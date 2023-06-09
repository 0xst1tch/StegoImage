# Image Steganography Tool

This Python program is an image steganography tool that allows you to hide text within an chosen image, and then later decode the hidden text from the introduced image.

## Features

- Hide text within an image without visually affecting the image
- Decode hidden text from an image

## Dependencies

- OpenCV
- tqdm
- NumPy

### Install dependencies 

```bash
pip install -r requirements.txt
```

## Limitations

- The maximum length of the text that can be hidden in the image is 65,535 characters.
- Works only with PNG image formats

## Usage

```bash
image_stego.py [-h] [--mode {encode,decode}] [--input_image INPUT_IMAGE] [--output_image OUTPUT_IMAGE] [--text_file TEXT_FILE]
```

### Encoding

To encode text into an image, you must provide an input image, output image, output gray image, and a text file containing the text to hide.

```bash
python image_stego.py --mode encode --input_image input.png --output_image output.png --output_gray_image gray_output.png --text_file text.txt
```

### Decoding

To decode text from an image, you must provide the input image and input gray image.

```bash
python image_stego.py --mode decode --input_image output.png --input_gray_image gray_output.png
```

After decoding, the program will prompt you to save the decoded text to a file or display it in the console.
