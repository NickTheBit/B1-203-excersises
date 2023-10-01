# Make can be used to easily allow us to rebuild and repackage our reports
# Bellow the requirements for each element are broken down and implemented

Covariances.png  GausianCountoursPlot.png  GausianCountoursPrevExercise.png  gausianMixPlot.png:
	python3 Lecture_5_203.py

gausianMixPlot.png: gausianMixturePlot.py
	python3 gausianMixturePlot.py

report.pdf: report.tex gausianMixPlot.png Covariances.png GausianCountoursPlot.png  GausianCountoursPrevExercise.png  gausianMixPlot.png
	pdflatex report.tex

Lecture_5_203.zip: report.pdf 
	mkdir Lecture_5_203
	cp report.pdf Lecture_5_203/
	cp Lecture_5_203.py Lecture_5_203/
	cp gmm_visualise.py Lecture_5_203/
	cp Exercise5_notebook_w_hints.ipynb Lecture_5_203/
	zip -r Lecture_5_203.zip Lecture_5_203/

clean:
	rm report.log report.aux report.out texput.log

wipe:
	rm -r Lecture_5_203
	rm Lecture_5_203.zip
	rm report.pdf
	rm *.png

all: Lecture_5_203.zip
