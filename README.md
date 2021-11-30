## c@

Fun university assignment where I got to write a very simple interpreter.

### Features

- variables
- math expressions
- printing values in different number bases

### Example Source Code

```
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
```

Output:

```
2
12
18
4
4
32
0x20
0b100000
```

## Dependencies

- Python >= 3.10 (uses the new pattern matching feature)
- pytest