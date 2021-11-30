from cat_lang import run_code


def test_example():
    # Example source code given by assignment hand out.
    source_code = """
    config dec
    print 1 + 1
    print 3 + 3 * 3
    print ( 3 + 3 ) * 3
    x = 2 - -2
    y = x
    z = y * ( 16 / ( y - 2 ) )
    print x
    print y
    print z
    config hex
    print z
    config bin
    print z
    """
    assert list(run_code(source_code)) == ["2", "12", "18", "4", "4", "32", "0x20", "0b100000"]

