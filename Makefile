FFMPEG_PREFIX ?= /usr

CXXFLAGS += -std=c++0x
CPPFLAGS += -I$(FFMPEG_PREFIX)/include  
CFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

fumbnailer: fumbnailer.o
	g++ $(LDFLAGS) -o $@ -L$(FFMPEG_PREFIX)/lib  $< -lavformat -lavcodec -lswscale -lavutil  -lm -lz -lbz2
