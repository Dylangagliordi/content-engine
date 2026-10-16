-- Where the raw video file for an external item lives on local disk
-- (populated by tiktok_reader.download_pending_videos). NULL = not yet
-- downloaded. Files live under data/raw_videos/ (gitignored).

ALTER TABLE external_content_items ADD COLUMN local_video_path TEXT;
