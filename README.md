# VideoChapterCreator
Automatic Video Chapter Creator with BERT NextSentencePrediction and KeySentenceGenerator

<!DOCTYPE html>
<html>
<body>

  <h2>What is Video Chapter Creator?</h1>
  <p>Video Chapter Creator automatically creates chapters of your video. Since video does not have the "contents" of books, it is difficult to skim through video to only extract necessary information efficiently. Video Chapter Creator works through uploaded video (currently only for education videos) and automatically creates timeline that can be used in YouTube. Timeline will be offered in text format, so just copy it and paste on the description box of the YouTube. YouTube will automatically change the timeline into video timestamp, which will be shown above the progress bar. <br> </p>

  <h2>How to use Video Chapter Creator?</h1>
  <p>Upload your video through "upload" button below, and click the "create" button. Timeline of the video will be offered ASAP. Just copy the timeline, and paste it on your YouTube. No worry, your video does not remain on our website. <br><br><br><br></p>

  <form method="GET" action="/create">
    <h4> copy and paste your youtube url </h4>
    <input type="text" name="url" value={{request.form.url}}>
    <button type="submit"> submit </button>
  </form>

</body>
</html>


