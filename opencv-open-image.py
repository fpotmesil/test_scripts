import cv2

def resize_with_aspect_ratio(
        image, 
        width=None, 
        height=None,
        inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))

    return cv2.resize(image, dim, interpolation=inter)


image = cv2.imread("C:/Users/fpotmesi/Pictures/family-pic.jpg")
resized_image = resize_with_aspect_ratio(image, width=400, height=600)
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Family", image)
cv2.imshow("Gray", gray_image)
cv2.imwrite("GrayFamily.jpg", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

