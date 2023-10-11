# Make can be used to easily allow us to rebuild and repackage our reports
# Bellow the requirements for each element are broken down and implemented

# Image recipies go here

# Report dependencies
report.pdf: report.tex 
	pdflatex report.tex
	rm report.aux
	rm report.log
	rm report.out

Lecture_7_203.zip: report.pdf Lecture_7_203.py
	mkdir Lecture_7_203
	cp report.pdf Lecture_7_203/
	cp Lecture_7_203.py Lecture_7_203/
	zip -r Lecture_7_203.zip Lecture_7_203/
	rm -r Lecture_7_203

wipe:
	rm Lecture_7_203.zip

all: Lecture_7_203.zip
