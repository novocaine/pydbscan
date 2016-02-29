CPPFLAGS := -O3 -march=native -stdlib=libc++ -std=c++1y

OBJS := main.o

dbscan: $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm dbscan $(OBJS)
