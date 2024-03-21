from pytube import YouTube

link = 'https://www.youtube.com/watch?v=oyxhHkOel2I'
YouTube(link).streams.first().download()
