def reverseArray(array):
    newArray = np.zeros_like(array)
    for index in range(len(array)):
        newArray[index] = array[len(array) - index - 1]
    return newArray