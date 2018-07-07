from PIL import Image
import face_recognition
import os


def file_path(path):
    files = os.listdir(path)
    filenames = []
    for file in files:
        filepath = path + '/' + file
        filenames.append(filepath)
    return filenames


def face_finding(ori_path, tar_path, index):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(ori_path)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    # Print the location of each face in this image
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        # Resize the image into 70*70 form
        pil_image = pil_image.resize((70, 70), Image.ANTIALIAS)
        pil_image.save(tar_path + "/" + "{:0>4d}".format(index) + ".jpg")

def main():
    path = 'files'
    ori_paths = file_path(path)
    target = 'facefiles'
    exist = os.path.exists(target)
    if not exist:
        os.makedirs(target)
    for i in range(len(ori_paths)):
        ori_paths[i] = ori_paths[i].strip()
        face_finding(ori_paths[i], target, i + 1)


if __name__ == '__main__':
    main()
