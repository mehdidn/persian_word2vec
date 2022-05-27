file1 = open('a.txt', 'r')
Lines = file1.readlines()

file2 = open('b.txt', 'w')

count = 0
# Strips the newline character
for line in Lines:
	count += 1
	edited_line = line.replace("  ", "\t")
	file2.writelines(edited_line)

file1.close()
file2.close()
