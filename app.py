import PySimpleGUI as sg
import os.path
from PIL import Image, ImageOps
import tempfile

from processing import *

sg.theme("DefaultNoMoreNagging")

op_button = ""
coldepth = 0
img_input = None

# Kolom Area No 1: Area open folder and select image
file_list_column = [
    [
        sg.Frame(
            "Image Information",
            [
                [sg.Text("Image Size : "), sg.Text(size=(13, 1), key="ImgSize")],
                [sg.Text("Color Depth : "), sg.Text(size=(13, 1), key="ImgColorDepth")],
            ],
            font=("OpenSans", 10),
        )
    ],
    [
        sg.Frame(
            "Choose image",
            [
                [
                    sg.In(size=(19, 1), enable_events=True, key="ImgFolder"),
                    sg.FolderBrowse(),
                ],
                [
                    sg.Listbox(
                        values=[], enable_events=True, size=(25, 10), key="ImgList"
                    )
                ],
            ],
            font=("OpenSans", 10),
        )
    ],
]
# Kolom Area No 2: Area viewer image input
image_viewer_column = [
    [sg.Text("Image Input :")],
    [sg.Text(size=(40, 1), key="FilepathImgInput")],
    [sg.Image(key="ImgInputViewer")],
]
# Kolom Area No 3: Area Image info dan Tombol list of processing
list_processing = [
    [
        sg.Text(
            "Prediction:",
        )
    ],
    [
        sg.Text(
            size=(0, 0),
            key="Prediction",
            font=("OpenSans", 100),
        )
    ],
    [
        sg.Button("Classification", size=(20, 1), key="Classification"),
    ],
]

# Kolom Area No 4: Area viewer image output
image_viewer_column2 = [
    [sg.Text("Image Input:")],
    [sg.Text(size=(40, 1), key="FilepathImgInput")],
    [sg.Image(key="ImgInputViewer")],
]

# Gabung Full layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column2),
        sg.VSeperator(),
        sg.Column(list_processing),
    ]
]

window = sg.Window("SIBI Classification Model CNN by Dharma Saputra", layout)
# Run the Event Loop

# nama image file temporary setiap kali processing output
filename_out = "out.png"

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Folder name was filled in, make a list of files in the folder
    if event == "ImgFolder":
        folder = values["ImgFolder"]

        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
        ]

        window["ImgList"].update(fnames)

    elif event == "ImgList":  # A file was chosen from the listbox
        try:
            filename = os.path.join(values["ImgFolder"], values["ImgList"][0])
            # print(values["ImgList"][0])
            window["FilepathImgInput"].update(filename)

            # Convert image to PNG format and resize to 64 x 64
            with Image.open(filename) as img:
                img_resized = img.resize((256, 256))
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_filename = temp_file.name
                img_resized.save(temp_filename)

            # Update image viewer
            window["ImgInputViewer"].update(filename=temp_filename)

            # Update image information
            img_input = Image.open(temp_filename)
            img_width, img_height = img_input.size
            window["ImgSize"].update(str(img_width) + " x " + str(img_height))
            coldepth = img_input.mode
            window["ImgColorDepth"].update(str(coldepth))
        except:
            pass
    elif event == "Classification":
        # try:
        prediction = Classification(values["ImgList"][0], values["ImgFolder"])
        print(prediction)
        window["Prediction"].update(prediction)
    # except:
    #     pass


# github.com/dharmasaputraa
