EXE = braid

CC = g++ -g
LIB = 

braid:	main.o braid.o
	$(CC) -o braid main.o braid.o $(LIB)

clean:
	rm -fr *~ *.o braid braid.tgz

dist:
	tar cvfz braid.tgz *.cc *.h Makefile CITATIONS.txt LICENSE.txt README SETUP

# dependencies
braid.o: braid.h
main.o: icosahedral_group.cc
