import { handleUpload } from '@vercel/blob/client';

export const config = { runtime: 'edge' };

export default async function handler(request) {
  return handleUpload({
    request,
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
}
