import cv2 as cv


def main():
    # Read the image from the file
    img = cv.imread("images/tennis_racket.jpg")
    print(img.size)
    cv.rectangle(img, (10, 10), (700, 500), (0, 0, 255), 2)
    # Display the image in a window
    cv.imshow("Display window", img)

    # Wait for a keystroke in the window
    k = cv.waitKey(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(k)
    cv.imwrite("images/saved.png", gray)


if __name__ == '__main__':
    main()
