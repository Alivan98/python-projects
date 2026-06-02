class Point:
    def __init__(self, x, y):
        """
        Initialize a point with coordinates (x, y).
        """
        self.__x = x
        self.__y = y
    
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"({self.__x}, {self.__y})"
    
    def __add__(self, other):
        """
        Add a point or a number to the current point.
        """
        if isinstance(other, Point):
            return Point(self.__x + other.__x, self.__y + other.__y)
        elif isinstance(other, (int, float)):
            return Point(self.__x + other, self.__y + other)
        else:
            raise TypeError(f"Unsupported operand types for +: 'Point' and {type(other).__name__}")
    
    def __radd__(self, other):
        """
        Add the current point to a number (for right-side addition).
        """
        return self.__add__(other)
    
    def __eq__(self, other):
        """
        Check if two points are equal.
        """
        if isinstance(other, Point):
            return self.__x == other.__x and self.__y == other.__y
        else:
            return False
    
    @property
    def x(self):
        """
        Get the x-coordinate of the point.
        """
        return self.__x
    
    @x.setter
    def x(self, value):
        """
        Set the x-coordinate of the point.
        """
        self.__x = value
    
    @property
    def y(self):
        """
        Get the y-coordinate of the point.
        """
        return self.__y
    
    @y.setter
    def y(self, value):
        """
        Set the y-coordinate of the point.
        """
        self.__y = value

def do_add(a, b):
    """
    Add two objects using the + operator.
    """
    return a + b

def main():
    """
    Main function to demonstrate the usage of the Point class and do_add function.
    """
    p1 = Point(1, 2)
    p2 = Point(3, 4)

    print("p1 + p2 =", p1 + p2)  # (4, 6)
    print("p1 + 10 =", p1 + 10)  # (11, 12)
    print("sum([p1, p2]) =", sum([p1, p2]))  # (4, 6)
    print("p1 == Point(1, 2) =", p1 == Point(1, 2))  # True
    print("p1 == 'Hello' =", p1 == "Hello")  # False

    print("do_add(Point(10, 20), Point(5, 5)) =", do_add(Point(10, 20), Point(5, 5)))  # (15, 25)

    # Accessing private fields via properties
    print("p1 x-coordinate:", p1.x)  # 1
    print("p1 y-coordinate:", p1.y)  # 2

    # Modifying private fields via setters
    p1.x = 100
    p1.y = 200
    print("Modified p1:", p1)  # (100, 200)

if __name__ == "__main__":
    main()