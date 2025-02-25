import React, { useEffect, useState } from 'react';
import axios from 'axios';

const LeadList = () => {
  const [strategy, setStrategy] = useState("");

  useEffect(() => {
    axios.get('/api/leads')
      .then(response => setStrategy(response.data.strategy))
      .catch(error => console.error('Error fetching strategy:', error));
  }, []);

  return (
    <div>
      <h1>Lead Agent AI - Business Strategy</h1>
      {strategy === "" ? (
        <p>Loading strategy...</p>
      ) : (
        <div>
          <textarea value={strategy} readOnly rows="15" cols="80" />
          <p>
            <a href="http://localhost:5000/api/download/doc1.txt" target="_blank" rel="noopener noreferrer">Download Doc1</a> | 
            <a href="http://localhost:5000/api/download/doc2.txt" target="_blank" rel="noopener noreferrer">Download Doc2</a> | 
            <a href="http://localhost:5000/api/download/doc3.txt" target="_blank" rel="noopener noreferrer">Download Doc3</a>
          </p>
        </div>
      )}
    </div>
  );
};

export default LeadList;
