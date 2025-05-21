class MyCustomError(Exception):
    pass
class ValueTooSmallError(MyCustomError):
    pass
class ValueTooLargeError(MyCustomError):
    pass


def check_value(number):
    if number < 5:
        raise ValueTooSmallError(f"The number {number} is to small")
    elif number > 15:
        raise ValueTooLargeError(f"The number {number} is too large")
    else:
        print(f"The number {number} is within the allowed range.")
        
try:
    user_input = int(input("Enter a number : "))
    check_value(user_input)
except ValueTooSmallError as e:
    print(e)
except ValueTooLargeError as e:
    print(e)
    
    
    
class NameTooShortError(Exception):
    pass
class NameTooLongError(Exception):
    pass

def check_name(name):
    if len(name) < 3:
        raise NameTooShortError("이름은 최소 3글자 이상이어야 합니다.")
    elif len(name) >= 5:
        raise NameTooLongError("이름은 5글자를 넘을 수 없습니다.")
    else:
        print(f"안녕하세요, {name}님!")
        
try:
    user_name = input("이름을 입력하세요 (3~4글자만 허용) : ")
    check_name(user_name)
except NameTooShortError as e:
    print(e)
except NameTooLongError as e:
    print(e)
