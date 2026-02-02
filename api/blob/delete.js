const { del } = require('@vercel/blob');

module.exports = async (req, res) => {
  try {
    if (req.method !== 'POST') {
      res.status(405).json({ error: 'method_not_allowed' });
      return;
    }
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', async () => {
      let payload = {};
      try {
        payload = JSON.parse(body || '{}');
      } catch (err) {
        payload = {};
      }
      const url = payload.url;
      if (!url) {
        res.status(400).json({ error: 'missing_url' });
        return;
      }
      await del(url);
      res.status(200).json({ ok: true });
    });
  } catch (err) {
    res.status(500).json({ error: 'blob_delete_failed', message: err.message });
  }
};
