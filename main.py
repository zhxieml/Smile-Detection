import face_finding
import os
import cv2
import lbp
import sys


def view_bar(num, mes):
    rate_num = num
    number = int(rate_num / 4)
    hashes = '=' * number
    spaces = ' ' * (25 - number)
    r = "\r\033[31;0m%s\033[0m：[%s%s]\033[32;0m%d%%\033[0m" % (mes, hashes, spaces, rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def main():
    path = 'files'
    ori_paths = face_finding.file_path(path)
    target = 'facefiles'
    exist = os.path.exists(target)
    if not exist:
        os.makedirs(target)

    i = 0
    for i in range(len(ori_paths)):
        ori_paths[i] = ori_paths[i].strip()
        face_finding.face_finding(ori_paths[i], target, i + 1)
        i += 1
        view_bar(i / 3999 * 100, 'Face Processing ')
    print("\nThe face files have been created in 'facefiles'.")

    face_name = face_finding.file_path('facefiles')
    final_result = []
    a = 0
    for name in face_name:
        image = cv2.imread(name)
        final_result.append(lbp.get_vector(image))
        a += 1
        view_bar(a / 3884 * 100, 'LBP Processing')

    b = 0
    print('')
    result_output = open('result_output.txt', 'w')
    for i in final_result:
        result_output.write(str(i) + '\n')
        b += 1
        view_bar(b / 3884 * 100, 'Result Collecting')

    result_output.close()

    print("\nThe lbp features have been collected in 'result_output.txt'")

if __name__ == '__main__':
    main()
