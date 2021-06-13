def GetOutput(input, filter, str, pad):
    return (input - filter + (2 * pad)) / str + 1

inputSize = int(input('input image size: '))
filterSize = int(input('filter size: '))
stride = int(input('stride: '))
padding = int(input('padding: '))

print(GetOutput(inputSize, filterSize, stride, padding))