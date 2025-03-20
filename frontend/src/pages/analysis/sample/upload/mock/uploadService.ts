export const beforeUpload = (file: File): boolean => {
  const isValidSize = file.size / 1024 / 1024 < 50;
  if (!isValidSize) {
    alert('文件大小不能超过50MB');
    return false;
  }
  return true;
};
