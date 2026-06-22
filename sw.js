self.addEventListener('install', e => {
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('message', async event => {
  if (event.data && event.data.type === 'SAVE_CHECKIN') {
    const cache = await caches.open('checkin-cache');
    await cache.put('/checkin-data', new Response(JSON.stringify(event.data.payload)));
  }
});
