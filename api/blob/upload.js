const { handleUpload } = require('@vercel/blob/client');

module.exports = async (req, res) => {
  await handleUpload({
    request: req,
    onBeforeGenerateToken: async (pathname) => {
      const allowed = ['.txt', '.md', '.docx'];
      const lower = pathname.toLowerCase();
      const ok = allowed.some((ext) => lower.endsWith(ext));
      if (!ok) {
        throw new Error('Chi ho tro .txt, .md, .docx');
      }
      return {
        allowedContentTypes: [
          'text/plain',
          'text/markdown',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        ],
        addRandomSuffix: false,
      };
    },
    onUploadCompleted: async () => {},
  });
};
