checkers.exe: nn.cpp constants.h
	g++ -std=c++11 -o nn.exe nn.cpp

debug: nn.cpp constants.h
	g++ -g -std=c++11 -o Debugnn.exe nn.cpp

clean:
	rm -f *.exe *.o *.stackdump *~

backup:
	test -d backups || mkdir backups
	cp *.cpp backups
	cp *.h backups
	cp Makefile backups