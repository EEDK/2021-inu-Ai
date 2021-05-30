import math as m


class point:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c


z1 = point(19.0, 7.0, 2)
z2 = point(6.6, 16.6, 3)
z3 = point(20.0, 15.5, 4)

z1_group = []
z2_group = []
z3_group = []


# 거리계산
def diffPos(x, y, new_x, new_y):
    diff_x = x - new_x
    diff_y = y - new_y

    return m.sqrt(pow(diff_x, 2) + pow(diff_y, 2))


# 군집 선택
def calc(x, y, i):
    z1_diff = diffPos(x, y, z1.x, z1.y)
    z2_diff = diffPos(x, y, z2.x, z2.y)
    z3_diff = diffPos(x, y, z3.x, z3.y)

    if z1_diff < z2_diff and z1_diff < z3_diff:
        print('x{}의 군집은 z1'.format(i + 1))
        z1_group.append(point(x, y, i + 1))
    elif z2_diff < z3_diff:
        print('x{}의 군집은 z2'.format(i + 1))
        z2_group.append(point(x, y, i + 1))
    else:
        print('x{}의 군집은 z3'.format(i + 1))
        z3_group.append(point(x, y, i + 1))


# 6.2
def calc_J():
    J = 0

    for i in range(len(z1_group)):
        z1_diff = diffPos(z1_group[i].x, z1_group[i].y, z1.x, z1.y)
        J += z1_diff

    for i in range(len(z2_group)):
        z2_diff = diffPos(z2_group[i].x, z2_group[i].y, z2.x, z2.y)
        J += z2_diff

    for i in range(len(z3_group)):
        z3_diff = diffPos(z3_group[i].x, z3_group[i].y, z3.x, z3.y)
        J += z3_diff

    return J


x = [18, 20, 20, 20, 5, 9, 6]
y = [5, 9, 14, 17, 15, 15, 20]

if __name__ == '__main__':
    N = 7
    for i in range(N):
        calc(x[i], y[i], i)

    temp_x = 0
    temp_y = 0
    if len(z1_group) != 0:
        for i in range(len(z1_group)):
            temp_x += z1_group[i].x
            temp_y += z1_group[i].y

        z1 = point(temp_x / len(z1_group), temp_y / len(z1_group), '0')

    temp_x = 0
    temp_y = 0
    if len(z2_group) != 0:
        for i in range(len(z2_group)):
            temp_x += z2_group[i].x
            temp_y += z2_group[i].y

        z2 = point(temp_x / len(z2_group), temp_y / len(z2_group), '0')

    temp_x = 0
    temp_y = 0
    if len(z3_group) != 0:
        for i in range(len(z3_group)):
            temp_x += z3_group[i].x
            temp_y += z3_group[i].y

        z3 = point(temp_x / len(z3_group), temp_y / len(z3_group), '0')

    print(z1.x, z1.y, z2.x, z2.y, z3.x, z3.y)

    J = calc_J()

    print(J)
