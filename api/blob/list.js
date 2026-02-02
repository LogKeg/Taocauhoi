const { list } = require('@vercel/blob');

module.exports = async (req, res) => {
  try {
    const prefix = typeof req.query.prefix === 'string' ? req.query.prefix : undefined;
    const limit = req.query.limit ? Number(req.query.limit) : 1000;
    const data = await list({ prefix, limit });
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: 'blob_list_failed', message: err.message });
  }
};
