# Make can be used to easily allow us to rebuild and repackage our reports
# Bellow the requirements for each element are broken down and implemented

pcaReduction.png ldaReduction.png: Lecture_6_203.py
	python3 Lecture_6_203.py

pcaReduction_to9.png ldaReduction_to9.png: reduction_to_9.py
	python3 reduction_to_9.py

report.pdf: report.tex pcaReduction.png ldaReduction.png pcaReduction_to9.png ldaReduction_to9.png
	pdflatex report.tex
	rm *.png

Lecture_6_203.zip: report.pdf Lecture_6_203.py
	mkdir Lecture_6_203
	cp report.pdf Lecture_6_203/
	cp Lecture_6_203.py Lecture_6_203/
	cp reduction_to_9.py Lecture_6_203/
	zip -r Lecture_6_203.zip Lecture_6_203/
	rm -r Lecture_6_203

wipe:
	rm Lecture_6_203.zip

all: Lecture_6_203.zip
