# importing library
import numpy

# initializing list
lst = [1, 7, 0, 6, 2, 5, 6]

# converting list to array
arr = numpy.array(lst)

# displaying list
print("List: ", lst)

# displaying array
print("Array: ", arr)

print("Array shape: ", arr.shape)
print("Transposed array: ", arr.reshape(-1,1))