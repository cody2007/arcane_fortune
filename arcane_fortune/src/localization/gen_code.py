# extracts the names of the keys in the `Generic text` block
# then prints the structure definitions, then prints the
# initializer code for those fields

end_delim1 = '// finished substitution 1'
end_delim2 = '// finished substitution 2'

###### build `Localization` structure definition and initializer from config/localization.txt
f = open('config/localization.txt').read()
lines = f.split('{\n')[2].split('}')[0].split('\n')

struct_txt = ''
initializer_txt = ''

for line in lines[:-1]:
        nm = line.strip().split(':')[0]
        struct_txt += 'pub ' + nm + ': String, '
        initializer_txt += nm + ': find_key("' + nm + '"), '

struct_txt += end_delim1
initializer_txt += end_delim2

f = open('localization/mod.rs').read()

######## replace structure definition
lines = f.split('pub Empty_txt: ')
pre_substitution = lines[0]
split2 = lines[1].split(end_delim1)

f = pre_substitution + struct_txt + split2[1]

####### replace 
lines = f.split('Empty_txt: find_key("Empty_txt")')
pre_substitution = lines[0]
split2 = lines[1].split(end_delim2)

f = pre_substitution + initializer_txt + split2[1]

file2 = open("localization/mod.rs","w")
file2.write(f);

