"""Monthly archiving system for paper data."""

import logging
import shutil
import tarfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Archiver:
    """Monthly archiver for paper data."""

    def __init__(self, data_dir: str = "./papers", archive_dir: str = "./archive"):
        """
        Initialize archiver.

        Args:
            data_dir: Directory containing daily JSON files
            archive_dir: Directory for storing compressed archives
        """
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized archiver: data_dir={data_dir}, archive_dir={archive_dir}"
        )

    def _get_archive_path(self, year: int, month: int) -> Path:
        """
        Get archive file path for a given year and month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Path to archive file
        """
        filename = f"papers_{year:04d}-{month:02d}.tar.gz"
        return self.archive_dir / filename

    def archive_month(self, year: int, month: int) -> bool:
        """
        Archive all daily files for a specific month.

        Args:
            year: Year to archive
            month: Month to archive (1-12)

        Returns:
            True if successful, False otherwise
        """
        # Find all daily files for the specified month
        daily_files = []
        pattern = f"papers_{year:04d}-{month:02d}-*.json"
        for file_path in self.data_dir.glob(pattern):
            if file_path.is_file():
                daily_files.append(file_path)

        if not daily_files:
            logger.info(f"No daily files found for {year:04d}-{month:02d}")
            return False

        # Create archive
        archive_path = self._get_archive_path(year, month)
        temp_archive_path = archive_path.with_suffix(".tmp.tar.gz")

        try:
            logger.info(
                f"Archiving {len(daily_files)} daily files for {year:04d}-{month:02d}"
            )

            # Create tar.gz archive
            with tarfile.open(temp_archive_path, "w:gz") as tar:
                for file_path in daily_files:
                    # Add file to archive with relative path
                    tar.add(file_path, arcname=file_path.name)

            # Move temporary archive to final location
            temp_archive_path.replace(archive_path)

            # Remove archived daily files
            for file_path in daily_files:
                file_path.unlink()

            logger.info(f"Archived {len(daily_files)} files to {archive_path}")
            return True

        except (IOError, OSError, tarfile.TarError) as e:
            logger.error(f"Failed to archive {year:04d}-{month:02d}: {e}")
            # Clean up temporary archive if it exists
            if temp_archive_path.exists():
                try:
                    temp_archive_path.unlink()
                except OSError:
                    pass
            return False

    def archive_old_data(self, retention_days: int = 30) -> List[Path]:
        """
        Archive data older than retention_days.

        Args:
            retention_days: Number of days to keep data before archiving

        Returns:
            List of archive paths created
        """
        cutoff_date = date.today() - timedelta(days=retention_days)
        logger.info(f"Archiving data older than {cutoff_date}")

        archives_created = []
        processed_months = set()

        # Find all daily files older than cutoff
        for file_path in self.data_dir.glob("papers_*.json"):
            try:
                # Extract date from filename
                date_str = file_path.stem.split("_")[1]
                file_date = date.fromisoformat(date_str)

                if file_date < cutoff_date:
                    year_month = (file_date.year, file_date.month)
                    if year_month not in processed_months:
                        # Archive entire month
                        if self.archive_month(file_date.year, file_date.month):
                            archives_created.append(
                                self._get_archive_path(file_date.year, file_date.month)
                            )
                        processed_months.add(year_month)
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid filename {file_path.name}: {e}")
                continue

        logger.info(f"Created {len(archives_created)} archives")
        return archives_created

    def list_archives(self) -> List[Path]:
        """
        List all archive files.

        Returns:
            List of archive paths, sorted by date (newest first)
        """
        archives = list(self.archive_dir.glob("papers_*.tar.gz"))
        archives.sort(reverse=True)  # Newest first
        return archives

    def extract_archive(
        self, archive_path: Path, extract_dir: Optional[Path] = None
    ) -> bool:
        """
        Extract an archive to a directory.

        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to (default: data_dir)

        Returns:
            True if successful, False otherwise
        """
        if extract_dir is None:
            extract_dir = self.data_dir

        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Extracting {archive_path} to {extract_dir}")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            return True
        except (IOError, OSError, tarfile.TarError) as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False

    def get_archive_info(self, archive_path: Path) -> Optional[dict]:
        """
        Get information about an archive.

        Args:
            archive_path: Path to archive file

        Returns:
            Dictionary with archive metadata, or None if error
        """
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                file_count = len(tar.getmembers())
                file_names = [member.name for member in tar.getmembers()]
                size = archive_path.stat().st_size

            return {
                "path": str(archive_path),
                "size_bytes": size,
                "file_count": file_count,
                "file_names": file_names,
                "modified": datetime.fromtimestamp(archive_path.stat().st_mtime),
            }
        except (IOError, OSError, tarfile.TarError) as e:
            logger.error(f"Failed to get archive info for {archive_path}: {e}")
            return None

    def _extract_date_from_archive_filename(self, filename: str) -> Optional[date]:
        """
        Extract date from archive filename.

        Args:
            filename: Archive filename (e.g., "papers_2020-01.tar.gz")

        Returns:
            Date object (first day of month) or None if parsing fails
        """
        try:
            # Remove .tar.gz extension and split
            if filename.endswith(".tar.gz"):
                filename = filename[:-7]  # Remove ".tar.gz"
            # Expected format: papers_YYYY-MM
            if not filename.startswith("papers_"):
                return None
            date_str = filename[7:]  # Remove "papers_"
            # Add day component for ISO format
            return date.fromisoformat(date_str + "-01")
        except (IndexError, ValueError) as e:
            logger.debug(f"Failed to extract date from filename {filename}: {e}")
            return None

    def cleanup_old_archives(self, max_archive_age_days: int = 365) -> int:
        """
        Delete archives older than specified age.

        Args:
            max_archive_age_days: Maximum age of archives to keep (in days)

        Returns:
            Number of archives deleted
        """
        cutoff_date = date.today() - timedelta(days=max_archive_age_days)
        deleted_count = 0

        for archive_path in self.list_archives():
            archive_date = self._extract_date_from_archive_filename(archive_path.name)
            if archive_date is None:
                logger.warning(f"Invalid archive filename {archive_path.name}")
                continue

            if archive_date < cutoff_date:
                archive_path.unlink()
                deleted_count += 1
                logger.info(f"Deleted old archive: {archive_path}")

        logger.info(f"Deleted {deleted_count} old archives")
        return deleted_count
