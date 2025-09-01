import base64
import os
import tempfile
from typing import List, Optional, Union
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from PyPDF2 import PdfReader, PdfWriter
from google.api_core.exceptions import TooManyRequests
from agno.tools import Toolkit
from agno.utils.log import logger
import io
import json

# Load environment variables
dotenv.load_dotenv()


class PdfConversionTools(Toolkit):
    """
    Agno Toolkit for converting PDFs to Markdown using Gemini LLM.
    
    This toolkit provides a tool to convert PDFs to Markdown with features like:
    - Page range selection (e.g., pages 5 to 8)
    - Specific page selection (e.g., pages 5, 7, 9)
    - Graceful handling of rate limiting (429 errors): Returns processed pages and informs user
    - Page-by-page processing to handle large PDFs
    """
    
    def __init__(self, **kwargs):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        super().__init__(
            name="pdf_to_markdown_tools",
            tools=[
                self.convert_pdf_to_markdown,
                self.convert_pdf_to_json,
            ],
            **kwargs
        )
    
    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        specific_pages: Optional[List[int]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Convert a PDF to Markdown format using Gemini LLM.
        
        Args:
            pdf_path (str): Path to the PDF file.
            start_page (Optional[int]): Starting page for range extraction (1-based).
            end_page (Optional[int]): Ending page for range extraction (1-based).
            specific_pages (Optional[List[int]]): List of specific page numbers to extract (1-based).
            output_file (Optional[str]): Path to save the output Markdown file.
            
        Returns:
            str: The converted Markdown content or a status message if partial due to errors.
        """
        try:
            logger.info(f"Starting PDF to Markdown conversion for: {pdf_path}")
            
            # Read the PDF
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            
            # Determine pages to process
            if specific_pages:
                pages_to_process = [p - 1 for p in specific_pages if 1 <= p <= num_pages]  # 0-based
            elif start_page and end_page:
                pages_to_process = list(range(max(0, start_page - 1), min(num_pages, end_page)))  # 0-based
            else:
                pages_to_process = list(range(num_pages))  # All pages, 0-based
            
            if not pages_to_process:
                return "No valid pages specified to process."
            
            markdown_outputs = []
            processed_pages = 0
            
            for page_num in pages_to_process:
                try:
                    logger.info(f"Processing page {page_num + 1} of {num_pages}")
                    
                    # Create a temporary single-page PDF
                    writer = PdfWriter()
                    writer.add_page(reader.pages[page_num])
                    
                    # In-memory PDF file
                    pdf_buffer = io.BytesIO()
                    writer.write(pdf_buffer)
                    pdf_buffer.seek(0)
                    b64_pdf = base64.b64encode(pdf_buffer.read()).decode()

                    # Compose a multimodal message for this page
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": "Please convert this page of the PDF to markdown. Return only markdown format. Do not hallucinate. Give proper structured markdown."},
                            {
                                "type": "file",
                                "source_type": "base64",
                                "mime_type": "application/pdf",
                                "data": b64_pdf,
                            },
                        ]
                    )

                    response = self.llm.invoke([message])
                    logger.info(f"Received response for page {page_num + 1}")
                    page_markdown = response.content.replace("```markdown", "").replace("```", "").strip()
                    
                    # Append to outputs
                    markdown_outputs.append(page_markdown)
                    processed_pages += 1
                    
                    # Clean up temporary file
                    pdf_buffer.close()
                
                except TooManyRequests as e:
                    # Handle 429 (rate limiting) or other 4xx errors
                    error_msg = f"Rate limit error (4xx) encountered after processing {processed_pages} pages: {str(e)}. Returning partial results up to page {page_num}."
                    logger.warning(error_msg)
                    full_markdown = "\n\n".join(markdown_outputs)
                    if output_file:
                        with open(output_file, "w") as f:
                            f.write(full_markdown)
                        return f"{error_msg} Partial Markdown saved to {output_file}."
                    return f"{error_msg}\n\nPartial Markdown:\n{full_markdown}"
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    # Continue to next page on other errors
            
            # Combine all page markdowns with page separators
            full_markdown = "\n\n".join(markdown_outputs)
            
            if output_file:
                with open(output_file, "w") as f:
                    f.write(full_markdown)
                return f"Full conversion complete. Markdown saved to {output_file}."
            
            return full_markdown
        
        except Exception as e:
            error_msg = f"Error in PDF to Markdown conversion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def convert_pdf_to_json(
        self,
        pdf_path: str,
        json_prompt_for_gemini: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Convert a PDF to JSON format using Gemini LLM.

        Args:
            pdf_path (str): Path to the PDF file.
            json_prompt_for_gemini (str): Prompt to guide the JSON conversion.
            output_file (Optional[str]): Path to save the output JSON file.

        Returns:
            str: The converted JSON content or a status message if partial due to errors.
        """
        try:
            logger.info(f"Starting PDF to JSON conversion for: {pdf_path}")

            pdf_bytes = open(pdf_path, 'rb').read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"JUST give a structured json response. Don't add any explanations or extra text in the response. There can be multiple pages. {json_prompt_for_gemini}"},
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type":"application/pdf",
                        "data": pdf_base64
                    }
                ]
            )
            resp = self.llm.invoke([message])
            json_response = resp.content.replace("```json", "").replace("```", "")
            
            if output_file:
                with open(output_file, "w") as f:
                    json_obj = json.loads(json_response)
                    print("✅ Gemini responded:", json_obj)
                    json.dump(json_obj, f, indent=2)
                return f"Full conversion complete. JSON saved to {output_file}."

            return json_response

        except Exception as e:
            error_msg = f"Error in PDF to JSON conversion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    
       
    def read_pdf(self, file_path: str, first_n_characters: int = None) -> str:
        """
        Read the first n characters of a file.
        If first_n_characters is None or greater than the length of the file, read the entire file.
        
        Args:
            file_path (str): Path to the file.
            first_n_characters (int, optional): Number of characters to read from the start of the file.
            
        Returns:
            str: Content of the file (up to first_n_characters if specified).
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            if first_n_characters is not None and first_n_characters > 0:
                return f.read(first_n_characters)
            else:
                return f.read()
