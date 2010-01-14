// Under the 
// GNU Affero General Public License version 3
// http://www.fsf.org/licensing/licenses/agpl-3.0.html
// (c) John Fremlin 2010

#include<vector>
#include<limits>
#include<iostream>
#include<map>

#include<stdarg.h>
#include<assert.h>
#include<stdio.h>

#include<opencv/highgui.h>
#include<opencv/cv.h>

#include<err.h>
extern "C" 
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

void e(const char*msg,...)
{
  va_list va;
  va_start(va,msg);
  verrx(1,msg,va);
  va_end(va);
}

struct frame_info
{
  uint64_t pos;//byte position in file
  uint64_t len;//bytes
  uint64_t usec;
  bool keyframe;
};

struct frameprocessor 
{
  virtual void process_frame(const frame_info&,AVFrame*,IplImage*){}
  virtual bool select_frame(frame_info&frame,std::string&tag){return false;}
  virtual ~frameprocessor(){}
};
struct frameprocessor_store:frameprocessor,std::vector<frame_info>
{
  virtual void process_frame(const frame_info&fi,AVFrame*,IplImage*){
    push_back(fi);
  }
};
struct frameprocessor_opencv:frameprocessor {
  CvHaarClassifierCascade*cascade;
  CvMemStorage*storage;
  
protected:
  typedef long double score_t;
  score_t score_frame(const frame_info&,AVFrame*,IplImage*opencv)
  {
    cvClearMemStorage( storage );
    CvSeq* faces = cvHaarDetectObjects(opencv, cascade, storage,
					1.05, 3, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
				       cvSize(16, 16));
    if(!faces)return worst_score();

    score_t score = 0;
    for(int i = 0; i < (faces ? faces->total : 0); ++i){
      CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
      score += (100 * r->width * r->height)/(opencv->width * opencv->height);
    }
    // XXX release faces
    return score;
  }
  
  static score_t worst_score()
  {
    return std::numeric_limits<score_t>::min();
  }
private:
  score_t _best_score;
  frame_info _best_fi;
public:

  void process_frame(const frame_info&fi,AVFrame*ffmpeg,IplImage*opencv){
    auto s = score_frame(fi,ffmpeg,opencv);
    if(s>_best_score){
      _best_score = s;
      _best_fi = fi;
    }
  }
  bool select_frame(frame_info&frame,std::string&tag){
    if(worst_score() == _best_score)return false;
    tag="opencv";
    frame=_best_fi;
    _best_score = worst_score();
    return true;
  }
  frameprocessor_opencv(const char*cascade_filename="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    :_best_score(worst_score())
  {
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_filename, 0, 0, 0 );
    if(!cascade)e("Unable to open OpenCV classifier cascade %s",cascade_filename);
    storage = cvCreateMemStorage(0);
    assert(storage);
  }
  ~frameprocessor_opencv()
  {
    /// XXX free storage and cascade
  }
};

struct frameprocessor_mest:frameprocessor_store
{
  bool select_frame(frame_info&best_fi,std::string&tag){
    if(empty())return false;

    tag="mest";
    typedef int64_t score_t;
    score_t worst_score = std::numeric_limits<score_t>::min();
    score_t best_score = worst_score;

    //only check the period after the keyframe
    for(auto i=begin();end()!=i;++i){
      auto j = i;
      ++j;
      if(end()==j)break;
      if(j->usec == i->usec)continue;

      score_t score = (1000000ll * (j->pos - (i->pos + i->len)))/(j->usec - i->usec);
      
      if(score>best_score){
	best_score = score;
	best_fi = *i;
      }
    }

    clear();
    return best_score != worst_score;
  }
};

inline
void opencv_from_ffmpeg_frame(SwsContext *sws,AVFrame*ffmpeg,IplImage*opencv)
{
  sws_scale(sws, ffmpeg->data, ffmpeg->linesize,
	    0, opencv->height, (uint8_t**)&opencv->imageData, &opencv->widthStep);
}

typedef bool(*frame_filter_t)(const frame_info&);

bool decode_and_convert_frame(AVCodecContext*avcc,AVPacket*pkt,SwsContext *sws,AVFrame*frame,IplImage*opencv)
{
  int got_picture;
  long len;
  
#ifdef FFMPEG_HAS_AVCODEC_VIDEO2
  len = avcodec_decode_video2(avcc,
			      frame, &got_picture,
					   pkt);
#else
  len = avcodec_decode_video(avcc,
			     frame, &got_picture,
			     pkt->data,pkt->size);
#endif
  
  if(!got_picture){
    warnx("video frame at byte %ld could not be decoded",(long)pkt->pos);
    return false;
  }
  if(len != pkt->size){
    warnx("video packet at byte %ld had length %ld but only %ld was decoded",(long)pkt->pos,
	  (long)pkt->size,
	  (long)len);
  }
      
  opencv_from_ffmpeg_frame(sws,frame,opencv);
  return true;
}

void process_video(const char*video_filename,frame_filter_t filter,frameprocessor**processors,const std::string& thumb_prefix,const std::string&thumb_suffix)
{
  AVFormatContext *ic;
  int err;
  AVPacket pkt1, *pkt = &pkt1;
  int vframeno=0;
  AVFrame *frame;
  SwsContext *sws = 0;
  IplImage*opencv = 0;
  unsigned video_stream_index;
  
  err = av_open_input_file(&ic, video_filename, 0, 0, 0);
  if(0>err)
    e("av_open_input_file(%s) returned %d",video_filename,err);
  err = av_find_stream_info(ic);
  if(0>err)
    e("av_find_stream_info on %s returned %d",video_filename,err);
  dump_format(ic,0,video_filename,0);

  for(int i=0;ic->nb_streams>i;++i){
    AVCodecContext* avcc = ic->streams[i]->codec;
    AVCodec *codec;
    
    if(CODEC_TYPE_VIDEO==avcc->codec_type){
	
      codec = avcodec_find_decoder(avcc->codec_id);
      if(!codec)
	e("unable to find decoder for stream %d (codec_id %d) in %s",i,(int)avcc->codec_id,video_filename);
      err = avcodec_open(avcc, codec);
      if(0>err)
	e("unable to open decoder for stream %d (codec_id %d) in %s",i,(int)avcc->codec_id,video_filename);
      assert(avcc->codec);
      if(sws)
	e("unable to handle multiple video streams in %s",video_filename);
      video_stream_index = i;
      sws = sws_getContext(
				 avcc->width, avcc->height,
				 avcc->pix_fmt,
				 avcc->width, avcc->height,
				 PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL); // XXX use BGR as OpenCV likes that (why?)
      assert(sws);

      opencv = cvCreateImage(cvSize(avcc->width,avcc->height), IPL_DEPTH_8U, 3);
      assert(opencv);      
    }
  }
  if(!sws)
    e("%s has no video stream", video_filename);
  
  frame = avcodec_alloc_frame();

  while((err=av_read_frame(ic,pkt))>=0) {
    AVStream*stream=ic->streams[pkt->stream_index];
    AVCodecContext*avcc=stream->codec;
    int64_t usec = (1000000l*(long)pkt->pts*(long)stream->time_base.num)
      /stream->time_base.den;

    if(CODEC_TYPE_VIDEO == avcc->codec_type){
      frame_info fi = {pkt->pos,pkt->size,usec,!!(PKT_FLAG_KEY&pkt->flags)};
      
      if(filter(fi)){
	decode_and_convert_frame(avcc,pkt,sws,frame,opencv);
	
	for(frameprocessor**i=processors;*i;++i)
	  (*i)->process_frame(fi,frame,opencv);
	++vframeno;
      }
    }
    av_free_packet(pkt);
  }

  if(AVERROR_EOF!=err)
    e("av_read_frame on %s returned %d after %d decoded video frames",video_filename,err,vframeno);

#ifdef SEEKING_NOT_BORKEN    
  for(frameprocessor**i=processors;*i;++i){
    frameprocessor*fp = *i;
    frame_info fi;
    std::string tag;
    while(fp->select_frame(fi,tag)){
      std::cout << tag << " at time " << fi.usec << std::endl;

      // XXX byte seeking seems not to work!
      // err = av_seek_frame(ic, video_stream_index, fi.pos, AVSEEK_FLAG_BYTE);
      
      err = av_seek_frame(ic, video_stream_index,(fi.usec*1000000ll)/AV_TIME_BASE,0);
      if(err)
	e("av_seek_frame on %s returned %d trying to seek to byte %ld at time %ld",video_filename,err,(long)fi.pos,(long)fi.usec);
      
      err=av_read_frame(ic,pkt);
      if(err)
	e("av_read_frame on %s returned %d after seeking to %ld",video_filename,err,(long)fi.pos);
      
      AVStream*stream=ic->streams[pkt->stream_index];
      AVCodecContext*avcc=stream->codec;
      decode_and_convert_frame(avcc,pkt,sws,frame,opencv);
      cvSaveImage((thumb_prefix+tag+thumb_suffix).c_str(),opencv);
      av_free_packet(pkt);
    }
  }
#else
  std::multimap<int64_t,std::string>wanted;
  for(frameprocessor**i=processors;*i;++i){
    frameprocessor*fp = *i;
    frame_info fi;
    std::string tag;
    while(fp->select_frame(fi,tag)){
      std::cout << tag << " at pos " << fi.pos << std::endl;
      wanted.insert(std::pair<int64_t,std::string>(fi.pos,tag));
    }
  }
  err = av_seek_frame(ic, -1, 0, AVSEEK_FLAG_BYTE);
  if(err)
    e("av_seek_frame on %s returned %d trying to seek to start",video_filename,err);
  
  for(auto next_wanted = wanted.begin();next_wanted != wanted.end();av_free_packet(pkt)) {
    err=av_read_frame(ic,pkt);
    if(err)
      e("av_read_frame on %s returned %d searching for wanted frames (next wanted pos %ld)",video_filename,err,(long)next_wanted->first);
    
    AVStream*stream=ic->streams[pkt->stream_index];
    AVCodecContext*avcc=stream->codec;
    int64_t usec = (1000000l*(long)pkt->pts*(long)stream->time_base.num)
      /stream->time_base.den;
    
    if(avcc->codec_type != CODEC_TYPE_VIDEO)
      continue;
    frame_info fi = {pkt->pos,pkt->size,usec,!!(PKT_FLAG_KEY&pkt->flags)};
    if(next_wanted->first < fi.pos)
      e("waiting for packet at %ld in %s but already at packet %ld",(long)next_wanted->first,video_filename,(long)fi.pos);
    
    while(next_wanted->first == fi.pos){
      decode_and_convert_frame(avcc,pkt,sws,frame,opencv);
      std::cout << next_wanted->second << " at time " << fi.usec << std::endl;
      cvSaveImage((thumb_prefix+next_wanted->second+thumb_suffix).c_str(),opencv);
      ++next_wanted;
      if(next_wanted == wanted.end())
	break;
    }
  }
#endif

  if(sws)
    sws_freeContext(sws);
  
  for(int i=0;ic->nb_streams>i;++i){
    AVCodecContext* avcc = ic->streams[i]->codec;
    // XXX crashes after seeking
    //avcodec_close(avcc);
  }
  av_close_input_file(ic);

  av_free(frame);

  if(opencv)
    cvReleaseImage(&opencv);
}

bool keyframe_filter(const frame_info&fi)
{
  return fi.keyframe;
}


void fumbnail(const char*video_filename,const char*thumbnail_filebase)
{
  frameprocessor_mest mest;
  frameprocessor_opencv opencv;
  frameprocessor* list[] =
    {
      &opencv,
      &mest,
     0};
  process_video(video_filename,keyframe_filter,list,thumbnail_filebase,".png");
}
int main(int argc,char*argv[])
{
  if(3 != argc)
    e("usage: videofile thumbnail");
  avcodec_register_all();
  av_register_all();

  fumbnail(argv[1],argv[2]);
  return 0;
}
