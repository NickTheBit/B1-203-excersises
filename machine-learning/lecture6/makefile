# Make can be used to easily allow us to rebuild and repackage our reports
# Bellow the requirements for each element are broken down and implemented

report.pdf: report.tex 
	pdflatex report.tex

Lecture_6_203.zip: report.pdf Lecture_6_203.py
	mkdir Lecture_6_203
	cp report.pdf Lecture_6_203/
	cp Lecture_6_203.py Lecture_6_203/
	cp gmm_visualise.py Lecture_6_203/
	cp Exercise5_notebook_w_hints.ipynb Lecture_6_203/
	zip -r Lecture_6_203.zip Lecture_6_203/

clean:
	rm report.log report.aux report.out texput.log

wipe:
	rm -r Lecture_6_203
	rm Lecture_6_203.zip
	rm report.pdf
	rm *.png

all: Lecture_6_203.zip
