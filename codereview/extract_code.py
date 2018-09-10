class FileHelper:

	def parse_fetched_file(self, file_name, data, sheet_name):
		valid_extensions = ['csv', 'xls', 'xlsx']
		extension = file_name.split('.')[-1]
		if extension == 'csv':
			return pd.read_csv(io.BytesIO(data.content))
		return pd.read_excel((io.BytesIO(data.content)), sheet_name=sheet_name)


class SftpHelper(FileHelper):
	def fetch_file_from_sftp(self, file_name, sheet_name = 0):
		sftp, transport = self.connect_to_sftp()
		remote_path = self.remote_dir + file_name
		data = io.BytesIO()
		sftp.getfo(remote_path, data, callback=None)
		file_df super(SftpHelper, self).parse_fetched_file(file_name, data, sheet_name)
		self.close_sftp_connection(sftp, transport)
		return file_df

class DropBoxHelper(FileHelper):
	def read_file_from_dropbox(self, file_name, sheet_name = 0):
		dbx = self.connect_to_dropbox()
		metadata, data = dbx.files_download(file_name)
		return super(DropBoxHelper, self).parse_fetched_file(file_name, data, sheet_name)

dbh = DropBoxHelper()
dbh.read_file_from_dropbox("test")