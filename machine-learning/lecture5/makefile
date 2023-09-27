# Integrate source files

report.pdf:
	pdflatex report.tex

Lecture_5_203.zip: report.pdf
	mkdir Lecture_5_203
	cp report.pdf Lecture_5_203/
	cp Lecture_5_203.py Lecture_5_203/
	zip -r Lecture_5_203.zip Lecture_5_203/

clean:
	rm report.log report.aux report.out texput.log

wipe:
	rm -r Lecture_5_203
	rm Lecture_5_203.zip
	rm report.pdf

all: Lecture_5_203.zip

