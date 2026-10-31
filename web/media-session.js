// Lock-screen and notification controls for the narration (Android Chrome), via the Media Session API.

const APP_ICON = { src: "/static/icons/icon-512.png", sizes: "512x512", type: "image/png" };
// Raster formats only: the notification needs a bitmap, so SVG figures fall back to the app icon.
const IMAGE_TYPES = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp" };
const supported = "mediaSession" in navigator;
let shown = "";

/** Wires the lock-screen buttons to `audio`; seekTo(seconds) clamps and seeks. */
export function initMediaSession(audio, seekTo, skipSeconds) {
  if (!supported) return;
  const actions = {
    play: () => audio.play().catch(() => {}), // a failure shows up through the element's "error" event
    pause: () => audio.pause(),
    seekbackward: (details) => seekTo(audio.currentTime - (details.seekOffset || skipSeconds)),
    seekforward: (details) => seekTo(audio.currentTime + (details.seekOffset || skipSeconds)),
    seekto: (details) => seekTo(details.seekTime),
  };
  for (const [action, handler] of Object.entries(actions)) {
    try {
      navigator.mediaSession.setActionHandler(action, handler);
    } catch {
      // this browser does not support the action
    }
  }
  for (const type of ["durationchange", "play", "pause", "seeked", "ratechange"]) {
    audio.addEventListener(type, () => updatePosition(audio));
  }
}

/** Title = the paper, artist = the narrated section, artwork = the figure's first raster image, else the app icon. */
export function setNowPlaying({ title, section, images }) {
  if (!supported) return;
  const image = images.find(imageType);
  const key = JSON.stringify([title, section, image]);
  if (key === shown) return;
  shown = key;
  navigator.mediaSession.metadata = new MediaMetadata({
    title,
    artist: section,
    album: "Paper Reader",
    artwork: [image ? { src: image, type: imageType(image) } : APP_ICON],
  });
}

function imageType(url) {
  return IMAGE_TYPES[url.split(".").pop().toLowerCase()];
}

/** The lock screen's progress bar. */
function updatePosition(audio) {
  if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
  navigator.mediaSession.setPositionState({
    duration: audio.duration,
    playbackRate: audio.playbackRate,
    position: Math.min(audio.currentTime, audio.duration),
  });
}
