from src import Picture

# Create a Picture object and define the base directory
picture = Picture("_example_data")

# Load pictures and save the average picture
directory_list = picture.get_directory_names()
for directory in directory_list:
    if directory.endswith("ms"):
        picture.load_pictures(directory)
        picture.save_picture(picture.average_picture(), "average", directory)

# Load the average picture and do some treatments on them
directory_name = "average"
picture.load_pictures(directory_name, extension=".png")
picture.traitement(
    treshold_min=70,
    treshold_max=255,
    smooth=30,
)